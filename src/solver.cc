// Created by jack on 9/25/19.
//

#include <algorithm>
#include <fstream>
#include <thread>
#include <vector>

#include <visualization_msgs/Marker.h>
#include <unsupported/Eigen/MatrixFunctions>
#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "eigen3/Eigen/Sparse"
#include "eigen3/Eigen/SparseQR"

#include <fstream>
#include "./kdtree.h"
#include "./line_extraction.h"
#include "./math_util.h"
#include "laser_scan_matcher/MatchLaserScans.h"
#include "nautilus/HitlSlamInputMsg.h"
#include "nautilus/WriteMsg.h"
#include "timer.h"

#include "./cimg_debug.h"
#include "./solver.h"

#define DEBUG true

using ceres::AutoDiffCostFunction;
using Eigen::Affine2f;
using Eigen::Matrix2f;
using Eigen::Rotation2D;
using Eigen::Vector2f;
using Eigen::Vector3f;
using laser_scan_matcher::MatchLaserScans;
using math_util::NormalsSimilar;
using slam_types::LidarFactor;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using std::vector;

using boost::math::chi_squared;
using boost::math::complement;
using boost::math::quantile;

namespace nautilus {

/*----------------------------------------------------------------------------*
 *                             PROBLEM BUILDING                               |
 *----------------------------------------------------------------------------*/

/* Solver takes in a ros node handle to be used for sending out debugging
 * information*/
// TODO: Upped the Scanmatcher resolution to 0.01 for ChiSquare.
Solver::Solver(ros::NodeHandle& n)
    : n_(n),
      vis_callback_(new VisualizationCallback(keyframes_, n_)),
      scan_matcher_(30, 2, 0.3, 0.01) {}

void Solver::AddSLAMNodeOdom(const SLAMNode2D& node,
                             const OdometryFactor2D& odom_factor_to_node) {
  CHECK_EQ(node.node_idx, odom_factor_to_node.pose_j);
  problem_.nodes.push_back(node);
  problem_.odometry_factors.push_back(odom_factor_to_node);
  initial_odometry_factors_.push_back(odom_factor_to_node);
  solution_.emplace_back(SLAMNodeSolution2D(node));
  vis_callback_->UpdateProblemAndSolution(
      node, &solution_, odom_factor_to_node);
}

void Solver::AddSlamNode(const SLAMNode2D& node) {
  problem_.nodes.push_back(node);
  solution_.emplace_back(SLAMNodeSolution2D(node));
  vis_callback_->UpdateProblemAndSolution(node, &solution_);
}

/*----------------------------------------------------------------------------*
 *                          SLAM SOLVING FUNCTIONS                            |
 *----------------------------------------------------------------------------*/

vector<SLAMNodeSolution2D> Solver::SolveSLAM() {
  // Setup ceres for evaluation of the problem.
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  options.callbacks.push_back(vis_callback_.get());
  double difference = 0;
  double last_difference = std::numeric_limits<double>::max();

  for (int i = 0; i < 5; i++) {
    vis_callback_->PubVisualization();
    sleep(1);
  }

  // While our solution moves more than the stopping_accuracy,
  // continue to optimize.
  CHECK_GT(config_.CONFIG_lidar_constraint_amount_min, 0);
  CHECK_GE(config_.CONFIG_lidar_constraint_amount_max,
           config_.CONFIG_lidar_constraint_amount_min);
  const size_t window_min = config_.CONFIG_lidar_constraint_amount_min;
  const size_t window_max = config_.CONFIG_lidar_constraint_amount_max;
  for (size_t window_size = window_min; window_size <= window_max;
       window_size++) {
    LOG(INFO) << "Using window size: " << window_size << std::endl;
    while (abs(difference - last_difference) >
           config_.CONFIG_accuracy_change_stop_threshold) {
      std::cout << "Solve diff " << abs(difference - last_difference)
                << " Target: " << config_.CONFIG_accuracy_change_stop_threshold
                << std::endl;
      vis_callback_->ClearNormals();
      last_difference = difference;
      difference = 0;
      ceres_information_.ResetProblem();
      // Add all the odometry constraints between our poses.
      AddOdomFactors(ceres_information_.problem.get(),
                     problem_.odometry_factors,
                     config_.CONFIG_translation_weight,
                     config_.CONFIG_rotation_weight);
      // For every SLAM node we want to optimize it against the past
      // lidar constraint amount nodes.
      for (size_t node_i_index = 0; node_i_index < problem_.nodes.size();
           node_i_index++) {
        std::mutex problem_mutex;
        const size_t node_j_start_idx =
            (node_i_index > window_size) ? (node_i_index - window_size) : 0;
#pragma omp parallel for default(none) shared(problem_mutex,      \
                                              vis_callback_,      \
                                              problem_,           \
                                              ceres_information_, \
                                              node_i_index,       \
                                              difference)
        for (size_t node_j_index = node_j_start_idx;
             node_j_index < node_i_index;
             node_j_index++) {
          ds::PointCorrespondences correspondence(solution_[node_i_index].pose,
                                                  solution_[node_j_index].pose,
                                                  node_i_index,
                                                  node_j_index);
          // Get the correspondences between these two poses.
          const double local_difference =
              GetPointCorrespondences(problem_.nodes[node_i_index].lidar_factor,
                                      problem_.nodes[node_j_index].lidar_factor,
                                      solution_[node_i_index].pose,
                                      solution_[node_j_index].pose,
                                      &correspondence) /
              problem_.nodes[node_j_index].lidar_factor.pointcloud.size();
          auto lidar_residual = residuals::LIDARPointBlobResidual::create(
              correspondence.source_points,
              correspondence.target_points,
              correspondence.source_normals,
              correspondence.target_normals);
          // Add the correspondences as constraints in the optimization problem.
          problem_mutex.lock();
          difference += local_difference;
          ceres::ResidualBlockId id =
              ceres_information_.problem->AddResidualBlock(
                  lidar_residual,
                  nullptr,
                  correspondence.source_pose,
                  correspondence.target_pose);
          ceres_information_.res_descriptors.emplace_back(
              node_i_index, node_j_index, id);
          problem_mutex.unlock();
        }
      }
      // Normalize the difference so it's an average over each node.
      difference /= problem_.nodes.size();
      AddHITLResiduals(ceres_information_.problem.get());
      ceres::Solve(options, ceres_information_.problem.get(), &summary);
    }
  }
  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i < 5; i++) {
    vis_callback_->PubVisualization();
    sleep(1);
  }
  return solution_;
}

void Solver::AddOdomFactors(ceres::Problem* ceres_problem,
                            vector<OdometryFactor2D> factors,
                            double trans_weight,
                            double rot_weight) {
  for (const OdometryFactor2D& odom_factor : factors) {
    CHECK_LT(odom_factor.pose_i, odom_factor.pose_j);
    CHECK_GT(solution_.size(), odom_factor.pose_i);
    CHECK_GT(solution_.size(), odom_factor.pose_j);
    ceres::ResidualBlockId id = ceres_problem->AddResidualBlock(
        residuals::OdometryResidual::create(
            odom_factor, trans_weight, rot_weight),
        NULL,
        solution_[odom_factor.pose_i].pose,
        solution_[odom_factor.pose_j].pose);
    ceres_information_.res_descriptors.emplace_back(
        odom_factor.pose_i, odom_factor.pose_j, id);
  }
  if (!solution_.empty()) {
    ceres_problem->SetParameterBlockConstant(solution_.front().pose);
  }
}

float ComputeNormalAndClosestWithDist(
    const LidarFactor& source_lidar,
    const LidarFactor& target_lidar,
    const Eigen::Vector2f& source_point,
    const Eigen::Vector2f& source_point_transformed,
    const float outlier_threshold,
    KDNodeValue<float, 2>* closest_target,
    KDNodeValue<float, 2>* source_point_with_normal) {
  vector<KDNodeValue<float, 2>> neighbors;
  target_lidar.pointcloud_tree->FindNeighborPoints(
      source_point_transformed, outlier_threshold, &neighbors);
  float found_dist = source_lidar.pointcloud_tree->FindNearestPoint(
      source_point, 0.1, source_point_with_normal);
  CHECK_EQ(found_dist, 0.0) << "Source point is not in KD Tree!\n";
  // Sort the target points by distance from the source point in the
  // target frame.
  std::sort(neighbors.begin(),
            neighbors.end(),
            [&source_point_transformed](const KDNodeValue<float, 2>& point_1,
                                        const KDNodeValue<float, 2>& point_2) {
              return (source_point_transformed - point_1.point).squaredNorm() <
                     (source_point_transformed - point_2.point).squaredNorm();
            });

  static constexpr double kMaxCosVal = cos(math_util::DegToRad(20.0));

  // For all target points, starting with the closest
  // see if any of them have a close enough normal to be
  // considered a match.
  for (const auto& current_target : neighbors) {
    if (NormalsSimilar(current_target.normal,
                       source_point_with_normal->normal,
                       kMaxCosVal)) {
      (*closest_target) = current_target;
      return (source_point_transformed - current_target.point).norm();
    }
  }
  return outlier_threshold;
}

// Source moves to target.
double Solver::GetPointCorrespondences(
    const LidarFactor& source_lidar,
    const LidarFactor& target_lidar,
    double* source_pose,
    double* target_pose,
    ds::PointCorrespondences* point_correspondences) {
  // Summed differences between point correspondences.
  double difference = 0.0;
  // Affine transformations from the two pose's reference frames.
  Affine2f source_to_world = PoseArrayToAffine(source_pose).cast<float>();
  Affine2f target_to_world = PoseArrayToAffine(target_pose).cast<float>();
  // Loop over all the points in the source pointcloud,
  // match each point to the closest point in the target pointcloud
  // who's normal is within a certain threshold.
  for (const Vector2f& source_point : source_lidar.pointcloud) {
    // Transform the source point to the target frame.
    const Vector2f source_point_transformed =
        target_to_world.inverse() * source_to_world * source_point;

    KDNodeValue<float, 2> closest_target;
    KDNodeValue<float, 2> source_point_with_normal;

    // For now we assume that a match is within 1/6 of the threshold.
    float dist =
        ComputeNormalAndClosestWithDist(source_lidar,
                                        target_lidar,
                                        source_point,
                                        source_point_transformed,
                                        config_.CONFIG_outlier_threshold / 6.0f,
                                        &closest_target,
                                        &source_point_with_normal);
    if (dist >= config_.CONFIG_outlier_threshold / 6.0f) {
      // Try again with full threshold.
      dist = ComputeNormalAndClosestWithDist(source_lidar,
                                             target_lidar,
                                             source_point,
                                             source_point_transformed,
                                             config_.CONFIG_outlier_threshold,
                                             &closest_target,
                                             &source_point_with_normal);
      if (dist >= config_.CONFIG_outlier_threshold) {
        difference += dist;
        continue;
      }
    }

    // Add the distance between the source point and it's matching target
    // point.
    difference += dist;
    // Add to the correspondence for returning.
    point_correspondences->source_points.push_back(source_point);
    point_correspondences->target_points.push_back(closest_target.point);
    point_correspondences->source_normals.push_back(
        source_point_with_normal.normal);
    point_correspondences->target_normals.push_back(closest_target.normal);
  }
  return difference;
}

vector<OdometryFactor2D> Solver::GetSolvedOdomFactors() {
  CHECK_GT(solution_.size(), 1);
  vector<OdometryFactor2D> factors;
  for (uint64_t index = 1; index < solution_.size(); index++) {
    // Get the change in translation.
    for (uint64_t prev_idx = std::max(
             (uint64_t)0, index - config_.CONFIG_lidar_constraint_amount_max);
         prev_idx < index;
         prev_idx++) {
      Vector2f prev_loc(solution_[prev_idx].pose[0],
                        solution_[prev_idx].pose[1]);
      Vector2f loc(solution_[index].pose[0], solution_[index].pose[1]);

      double rot_change =
          solution_[index].pose[2] - solution_[prev_idx].pose[2];
      Vector2f trans_change = loc - prev_loc;
      factors.emplace_back(prev_idx, index, trans_change, rot_change);
    }
  }
  return factors;
}

OdometryFactor2D Solver::GetTotalOdomChange(
    const std::vector<OdometryFactor2D>& factors) {
  Vector2f init_trans(0, 0);
  OdometryFactor2D factor(0, factors.size(), init_trans, 0);
  for (const auto& curr_factor : factors) {
    factor.translation += curr_factor.translation;
    factor.rotation += curr_factor.rotation;
    factor.rotation = math_util::AngleMod(factor.rotation);
  }
  return factor;
}

/*----------------------------------------------------------------------------*
 *                        HUMAN-IN-THE-LOOP LOOP CLOSURE                      |
 *----------------------------------------------------------------------------*/

vector<ds::LineSegment<float>> LineSegmentsFromHitlMsg(
    const HitlSlamInputMsg& msg) {
  Vector2f start_a(msg.line_a_start.x, msg.line_a_start.y);
  Vector2f end_a(msg.line_a_end.x, msg.line_a_end.y);
  Vector2f start_b(msg.line_b_start.x, msg.line_b_start.y);
  Vector2f end_b(msg.line_b_end.x, msg.line_b_end.y);
  vector<ds::LineSegment<float>> lines;
  lines.emplace_back(start_a, end_a);
  lines.emplace_back(start_b, end_b);
  return lines;
}

ds::HitlLCConstraint Solver::GetRelevantPosesForHITL(
    const HitlSlamInputMsg& hitl_msg) {
  // Linearly go through all poses
  // Go through all points and see if they lie on either of the two lines.
  const vector<ds::LineSegment<float>> lines =
      LineSegmentsFromHitlMsg(hitl_msg);
  ds::HitlLCConstraint hitl_constraint(lines[0], lines[1]);
  for (size_t node_idx = 0; node_idx < problem_.nodes.size(); node_idx++) {
    vector<Vector2f> points_on_a;
    vector<Vector2f> points_on_b;
    double* pose_ptr = solution_[node_idx].pose;
    Affine2f node_to_world =
        PoseArrayToAffine(&pose_ptr[2], &pose_ptr[0]).cast<float>();
    for (const Vector2f& point :
         problem_.nodes[node_idx].lidar_factor.pointcloud) {
      Vector2f point_transformed = node_to_world * point;
      if (DistanceToLineSegment(point_transformed, lines[0]) <=
          config_.CONFIG_hitl_line_width) {
        points_on_a.push_back(point);
      } else if (DistanceToLineSegment(point_transformed, lines[1]) <=
                 config_.CONFIG_hitl_line_width) {
        points_on_b.push_back(point);
      }
    }
    if (points_on_a.size() >=
        static_cast<size_t>(config_.CONFIG_hitl_pose_point_threshold)) {
      hitl_constraint.line_a_poses.emplace_back(node_idx, points_on_a);
    } else if (points_on_b.size() >=
               static_cast<size_t>(config_.CONFIG_hitl_pose_point_threshold)) {
      hitl_constraint.line_b_poses.emplace_back(node_idx, points_on_b);
    }
  }
  return hitl_constraint;
}

void Solver::AddHITLResiduals(ceres::Problem* problem) {
  for (ds::HitlLCConstraint& constraint : hitl_constraints_) {
    for (const ds::LCPose& a_pose : constraint.line_a_poses) {
      const vector<Vector2f>& pointcloud_a = a_pose.points_on_feature;
      CHECK_LT(a_pose.node_idx, solution_.size());
      problem->AddResidualBlock(residuals::PointToLineResidual::create(
                                    constraint.line_a, pointcloud_a),
                                nullptr,
                                solution_[a_pose.node_idx].pose,
                                constraint.chosen_line_pose);
    }
    for (const ds::LCPose& b_pose : constraint.line_b_poses) {
      const vector<Vector2f>& pointcloud_b = b_pose.points_on_feature;
      CHECK_LT(b_pose.node_idx, solution_.size());
      problem->AddResidualBlock(residuals::PointToLineResidual::create(
                                    constraint.line_a, pointcloud_b),
                                nullptr,
                                solution_[b_pose.node_idx].pose,
                                constraint.chosen_line_pose);
    }
  }
}

void Solver::HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr) {
  problem_.odometry_factors = GetSolvedOdomFactors();
  const HitlSlamInputMsg hitl_msg = *hitl_ptr;
  // Get the poses that belong to this input.
  const ds::HitlLCConstraint colinear_constraint =
      GetRelevantPosesForHITL(hitl_msg);
  std::cout << "Found " << colinear_constraint.line_a_poses.size()
            << " poses for the first line." << std::endl;
  std::cout << "Found " << colinear_constraint.line_b_poses.size()
            << " poses for the second line." << std::endl;
  vis_callback_->AddConstraint(colinear_constraint);
  for (int i = 0; i < 5; i++) {
    vis_callback_->PubVisualization();
    sleep(1);
  }
  hitl_constraints_.push_back(colinear_constraint);
  vis_callback_->PubVisualization();
  // Resolve the initial problem with extra pointcloud residuals between these
  // loop closed points.
  // TODO: Find a better way to set these up.
  //  translation_weight_ = lc_translation_weight_;
  //  rotation_weight_ = lc_rotation_weight_;
  std::cout << "Solving problem with HITL constraints..." << std::endl;
  SolveSLAM();
  // TODO: This is giving worse results.
  problem_.odometry_factors = initial_odometry_factors_;
  std::cout << "Solving problem with initial odometry constraints..."
            << std::endl;
  SolveSLAM();
  std::cout << "Waiting for Loop Closure input." << std::endl;
}

/*----------------------------------------------------------------------------*
 *                            MISC. SOLVER CALLBACKS                          |
 *----------------------------------------------------------------------------*/

void Solver::LoadSLAMSolution(const std::string& poses_path) {
  std::map<double, Vector3f> poses;
  std::ifstream poses_file(poses_path);
  double timestamp;
  float pose_x, pose_y, theta;
  while (poses_file >> timestamp >> pose_x >> pose_y >> theta) {
    poses[timestamp] = Vector3f(pose_x, pose_y, theta);
  }
  poses_file.close();
  std::cout << "Finished loading solution file." << std::endl;
  for (auto& s : solution_) {
    std::stringstream ss;
    ss << std::fixed << s.timestamp;
    double timestamp = std::stod(ss.str());
    if (poses.find(timestamp) != poses.end()) {
      s.pose[0] = poses[timestamp][0];
      s.pose[1] = poses[timestamp][1];
      s.pose[2] = poses[timestamp][2];
    } else {
      std::cout << "Unable to find solution for timestamp " << timestamp
                << "\n";
    }
  }

  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i < 5; i++) {
    vis_callback_->PubVisualization();
    sleep(1);
  }
}

void Solver::WriteCallback(const WriteMsgConstPtr& msg) {
  if (config_.CONFIG_pose_output_file == "") {
    std::cout << "No output file specified, not writing!" << std::endl;
    return;
  }
  std::cout << "Writing Poses" << std::endl;
  std::ofstream output_file;
  output_file.open(config_.CONFIG_pose_output_file);
  for (const SLAMNodeSolution2D& sol_node : solution_) {
    output_file << std::fixed << sol_node.timestamp << " " << sol_node.pose[0]
                << " " << sol_node.pose[1] << " " << sol_node.pose[2]
                << std::endl;
  }
  output_file.close();
}

vector<Vector2f> TransformPointcloud(double* pose,
                                     const vector<Vector2f>& pointcloud) {
  vector<Vector2f> pcloud;
  pcloud.reserve(pointcloud.size());
  Eigen::Affine2f trans = PoseArrayToAffine(pose).cast<float>();
  for (const Vector2f& p : pointcloud) {
    pcloud.push_back(trans * p);
  }
  return pcloud;
}

void Solver::Vectorize(const WriteMsgConstPtr& msg) {
  std::cout << "Vectorizing" << std::endl;
  vector<Vector2f> whole_pointcloud;
  for (const SLAMNode2D& n : problem_.nodes) {
    vector<Vector2f> pc = n.lidar_factor.pointcloud;
    pc = TransformPointcloud(solution_[n.node_idx].pose, pc);
    whole_pointcloud.insert(whole_pointcloud.begin(), pc.begin(), pc.end());
  }

  auto lines = VectorMaps::ExtractLines(whole_pointcloud);
  // --- Visualize ---
  visualization_msgs::Marker line_mark;
  gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                gui_helpers::Color4f::kWhite,
                                0.05,
                                0.00,
                                0.00,
                                &line_mark);
  ros::Publisher lines_pub =
      n_.advertise<visualization_msgs::Marker>("/debug_lines", 10);
  for (const auto& line : lines) {
    Vector3f line_start(line.start_point.x(), line.start_point.y(), 0.0);
    Vector3f line_end(line.end_point.x(), line.end_point.y(), 0.0);
    gui_helpers::AddLine(
        line_start, line_end, gui_helpers::Color4f::kWhite, &line_mark);
  }

  std::cout << "Created map: Pointcloud size: " << whole_pointcloud.size()
            << "\tLines size: " << lines.size() << std::endl;

  if (config_.CONFIG_map_output_file != "") {
    std::cout << "Writing map to file..." << std::endl;
    std::ofstream output_file;
    output_file.open(config_.CONFIG_map_output_file);
    for (auto line : lines) {
      output_file << line.start_point.x() << "," << line.start_point.y() << ","
                  << line.end_point.x() << "," << line.end_point.y()
                  << std::endl;
    }
    output_file.close();
  }

  std::cout << "Publishing map..." << std::endl;
  for (int i = 0; i < 5; i++) {
    lines_pub.publish(line_mark);
    sleep(1);
  }
}
}  // namespace nautilus