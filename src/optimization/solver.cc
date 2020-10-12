// Created by jack on 9/25/19.
//

#include "./solver.h"

#include <visualization_msgs/Marker.h>

#include <algorithm>
#include <fstream>
#include <thread>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

#include "../util/kdtree.h"
#include "../util/math_util.h"
#include "../util/timer.h"
#include "../visualization/cimg_debug.h"
#include "./data_structures.h"
#include "./line_extraction.h"
#include "./slam_residuals.h"
#include "Eigen/Geometry"
#include "ceres/ceres.h"
#include "laser_scan_matcher/MatchLaserScans.h"
#include "local_uncertainty_estimator/EstimateLocalUncertainty.h"
#include "nautilus/HitlSlamInputMsg.h"
#include "nautilus/WriteMsg.h"

#define DEBUG true

namespace nautilus {

using boost::math::chi_squared;
using ceres::AutoDiffCostFunction;
using Eigen::Affine2f;
using Eigen::Matrix2f;
using Eigen::Rotation2D;
using Eigen::Vector2f;
using Eigen::Vector3f;
using laser_scan_matcher::MatchLaserScans;
using local_uncertainty_estimator::EstimateLocalUncertainty;
using math_util::NormalsSimilar;
using nautilus::HitlSlamInputMsg;
using nautilus::HitlSlamInputMsgConstPtr;
using nautilus::WriteMsgConstPtr;
using slam_types::LidarFactor;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using std::pair;
using std::vector;

/*----------------------------------------------------------------------------*
 *                             PROBLEM BUILDING                               |
 *----------------------------------------------------------------------------*/

/* Solver takes in a ros node handle to be used for sending out debugging
 * information*/
// TODO: Upped the Scanmatcher resolution to 0.01 for ChiSquare.
Solver::Solver(ros::NodeHandle &n) : n_(n), scan_matcher(30, 2, 0.3, 0.01) {
  matcher_client = n_.serviceClient<MatchLaserScans>("match_laser_scans");
  local_uncertainty_client =
      n_.serviceClient<EstimateLocalUncertainty>("estimate_local_uncertainty");
  vis_callback_ = std::unique_ptr<VisualizationCallback>(
      new VisualizationCallback(keyframes, n_));
}

void Solver::AddSLAMNodeOdom(SLAMNode2D &node,
                             OdometryFactor2D &odom_factor_to_node) {
  CHECK_EQ(node.node_idx, odom_factor_to_node.pose_j);
  problem_.nodes.push_back(node);
  problem_.odometry_factors.push_back(odom_factor_to_node);
  initial_odometry_factors.push_back(odom_factor_to_node);
  SLAMNodeSolution2D sol_node(node);
  solution_.push_back(sol_node);
  vis_callback_->UpdateProblemAndSolution(node, &solution_,
                                          odom_factor_to_node);
}

void Solver::AddSlamNode(SLAMNode2D &node) {
  problem_.nodes.push_back(node);
  SLAMNodeSolution2D sol_node(node);
  solution_.push_back(sol_node);
  vis_callback_->UpdateProblemAndSolution(node, &solution_);
}

/*----------------------------------------------------------------------------*
 *                          SLAM SOLVING FUNCTIONS                            |
 *----------------------------------------------------------------------------*/

// Solves the pose-only version of slam (no lidar factors)
vector<SLAMNodeSolution2D> Solver::SolvePoseSLAM() {
  // ceres::Solver::Options options;
  // ceres::Solver::Summary summary;
  // options.linear_solver_type = ceres::SPARSE_SCHUR;
  // options.minimizer_progress_to_stdout = false;
  // options.num_threads =
  // static_cast<int>(std::thread::hardware_concurrency());
  // options.callbacks.push_back(vis_callback_.get());
  // for (int64_t i = 0; i < 20; i++) {
  //   vis_callback_->ClearNormals();
  //   ceres_information.ResetProblem();
  //   // Add all the odometry constraints between our poses.
  //   AddOdomFactors(ceres_information.problem.get(), GetSolvedOdomFactors(),
  //                   SolverConfig::CONFIG_translation_weight,
  //                   SolverConfig::CONFIG_rotation_weight);
  //   AddResidualsForAutoLC(ceres_information.problem.get(), false);
  //   ceres::Solve(options, ceres_information.problem.get(), &summary);
  // }

  // Apply Loop closures to current solution
  // This is COP SLAM
  for (auto constraint : auto_lc_constraints_) {
    uint64_t i = constraint.node_a->node_idx;
    uint64_t j = constraint.node_b->node_idx;
    SLAMNodeSolution2D sol_i = solution_[constraint.node_a->node_idx];
    SLAMNodeSolution2D sol_j = solution_[constraint.node_b->node_idx];
    Eigen::Affine2d A_Mi = Eigen::Translation2d(sol_i.pose[0], sol_i.pose[1]) *
                           Eigen::Rotation2Dd(sol_i.pose[2]);
    Eigen::Affine2d A_Mj = Eigen::Translation2d(sol_j.pose[0], sol_j.pose[1]) *
                           Eigen::Rotation2Dd(sol_j.pose[2]);
    Eigen::Affine2d A_Mj_star =
        A_Mi * Eigen::Affine2d(
                   Eigen::Translation2d(constraint.relative_transformation[0],
                                        constraint.relative_transformation[1]) *
                   Eigen::Rotation2Dd(constraint.relative_transformation[2]));

    std::cout << "Mi" << std::endl;
    std::cout << A_Mi.matrix() << std::endl;
    std::cout << "Mj" << std::endl;
    std::cout << A_Mj.matrix() << std::endl;
    std::cout << "Mj_star" << std::endl;
    std::cout << A_Mj_star.matrix() << std::endl;

    // std::vector<OdometryFactor2D> factors =
    // GetSolvedOdomFactorsBetweenNodes(i, j); OdometryFactor2D chainFactor =
    // GetTotalOdomChange(factors); Eigen::Affine2d A_chain =
    // Eigen::Translation2f(chainFactor.translation) *
    // Eigen::Rotation2Df(chainFactor.rotation);

    // // Validated that A_Mi * A_chain = A_Mj, as expected!
    // std::cout << "Composite" << std::endl;
    // std::cout << (A_Mi * A_chain).matrix() << std::endl;

    // Now do COP-SLAM
    uint64_t N = j - i;

    Eigen::Affine2d DeltaA = A_Mj_star.inverse() * A_Mj;
    std::cout << "DELTA A\t" << DeltaA.matrix() << std::endl;

    Eigen::Matrix3d deltaAMat = DeltaA.matrix().pow(1.0 / N);

    std::cout << "dalta A\t" << deltaAMat.matrix() << std::endl;

    // update poses involved in LC
    for (uint64_t k = 1; k < N; k++) {
      Eigen::Matrix3d poseUpdateMat = DeltaA.matrix().pow((double)k / N);
      Eigen::Affine2d poseUpdate(poseUpdateMat);
      solution_[i + k].pose[0] += poseUpdate.translation().x();
      solution_[i + k].pose[1] += poseUpdate.translation().y();
      solution_[i + k].pose[2] +=
          Eigen::Rotation2Dd(poseUpdate.rotation()).angle();
    }

    // Update all subsequent poses
    for (uint64_t k = 0; k < solution_.size() - j; k++) {
      solution_[j + k].pose[0] += DeltaA.translation().x();
      solution_[j + k].pose[1] += DeltaA.translation().y();
      solution_[j + k].pose[2] += Eigen::Rotation2Dd(DeltaA.rotation()).angle();
    }

    // compute weights;
    // for now, all 1?
    /*
    std::vector<double> weights;
    for(uint64_t k = 0; k < N; k++) {
      weights.push_back(0.1);
    }

    Eigen::Matrix3f jarg = (A_Mj.inverse() * A_Mj_star).matrix();
    Eigen::Matrix3f jarg_log = jarg.log().matrix();

    printf("computed jargs\n");
    std::cout << jarg << std::endl;
    printf("LOG\n");
    std::cout << jarg_log << std::endl;

    std::vector<Affine2f> updates;
    for(uint64_t k = 0; k < N; k++) {
      double alpha_1;
      double alpha_2;
      if (k > 0) {
        alpha_1 = std::accumulate(weights.begin(), weights.begin() + k - 1,
    0.0); alpha_2 = std::accumulate(weights.begin(), weights.begin() + k, 0.0);
      } else {
        alpha_1 = 0;
        alpha_2 = weights[0];
      }

      Affine2f J_1 = Eigen::Affine2f(((alpha_1 * jarg_log).exp()).matrix());
      Affine2f J_2 = Eigen::Affine2f(((alpha_2 * jarg_log).exp()).matrix());
      // Compute local update (8)
      Affine2f update_local = J_1.inverse() * J_2;
      // Compute distributed update (14)
      Eigen::Affine2f A_k = Eigen::Translation2f(solution_[i + k].pose[0],
    solution_[i + k].pose[1]) * Eigen::Rotation2Df(solution_[i + k].pose[2]);
      Affine2f update_distributed = A_k.inverse() * A_Mj_star * update_local *
    A_Mj_star.inverse() * A_k;

      std::cout << "UPDATE" << update_distributed.matrix() << std::endl;

      solution_[i+k].pose[0] += update_distributed.translation().x();
      solution_[i+k].pose[1] += update_distributed.translation().y();
      solution_[i+k].pose[2] +=
    Eigen::Rotation2Df(update_distributed.rotation()).angle();
    }*/
  }

  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i < 5; i++) {
    vis_callback_->PubVisualization();
    sleep(1);
  }

  return solution_;
}

vector<SLAMNodeSolution2D> Solver::SolveSLAM() {
  std::cout << "Optimizing, Trans Weight: "
            << SolverConfig::CONFIG_translation_weight
            << " Rotation Weight: " << SolverConfig::CONFIG_rotation_weight
            << std::endl;
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
  for (int64_t window_size = SolverConfig::CONFIG_lidar_constraint_amount_min;
       window_size <= SolverConfig::CONFIG_lidar_constraint_amount_max;
       window_size++) {
    LOG(INFO) << "Using window size: " << window_size << std::endl;
    while (abs(difference - last_difference) >
           SolverConfig::CONFIG_accuracy_change_stop_threshold) {
      vis_callback_->ClearNormals();
      last_difference = difference;
      difference = 0;
      ceres_information.ResetProblem();
      // Add all the odometry constraints between our poses.
      AddOdomFactors(ceres_information.problem.get(), problem_.odometry_factors,
                     SolverConfig::CONFIG_translation_weight,
                     SolverConfig::CONFIG_rotation_weight);
      // For every SLAM node we want to optimize it against the past
      // lidar constraint amount nodes.
      for (size_t node_i_index = 0; node_i_index < problem_.nodes.size();
           node_i_index++) {
        std::mutex problem_mutex;
#pragma omp parallel for
        for (size_t node_j_index =
                 std::max((int64_t)(node_i_index)-window_size, 0l);
             node_j_index < node_i_index; node_j_index++) {
          PointCorrespondences correspondence(solution_[node_i_index].pose,
                                              solution_[node_j_index].pose,
                                              node_i_index, node_j_index);
          // Get the correspondences between these two poses.
          difference +=
              GetPointCorrespondences(problem_.nodes[node_i_index].lidar_factor,
                                      problem_.nodes[node_j_index].lidar_factor,
                                      solution_[node_i_index].pose,
                                      solution_[node_j_index].pose,
                                      &correspondence) /
              problem_.nodes[node_j_index].lidar_factor.pointcloud.size();
          // Add the correspondences as constraints in the optimization problem.
          problem_mutex.lock();
          vis_callback_->UpdateLastCorrespondence(correspondence);
          ceres::ResidualBlockId id =
              ceres_information.problem->AddResidualBlock(
                  LIDARPointBlobResidual::create(correspondence.source_points,
                                                 correspondence.target_points,
                                                 correspondence.source_normals,
                                                 correspondence.target_normals),
                  NULL, correspondence.source_pose, correspondence.target_pose);
          ceres_information.res_descriptors.emplace_back(node_i_index,
                                                         node_j_index, id);
          problem_mutex.unlock();
        }
      }
      // Normalize the difference so it's an average over each node.
      difference /= problem_.nodes.size();
      AddHITLResiduals(ceres_information.problem.get());
      ceres::Solve(options, ceres_information.problem.get(), &summary);
    }
  }
  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i < 5; i++) {
    vis_callback_->PubVisualization();
    sleep(1);
  }
  return solution_;
}

void Solver::AddOdomFactors(ceres::Problem *ceres_problem,
                            vector<OdometryFactor2D> factors,
                            double trans_weight, double rot_weight) {
  for (const OdometryFactor2D &odom_factor : factors) {
    CHECK_LT(odom_factor.pose_i, odom_factor.pose_j);
    CHECK_GT(solution_.size(), odom_factor.pose_i);
    CHECK_GT(solution_.size(), odom_factor.pose_j);
    ceres::ResidualBlockId id = ceres_problem->AddResidualBlock(
        OdometryResidual::create(odom_factor, trans_weight, rot_weight), NULL,
        solution_[odom_factor.pose_i].pose, solution_[odom_factor.pose_j].pose);
    ceres_information.res_descriptors.emplace_back(odom_factor.pose_i,
                                                   odom_factor.pose_j, id);
  }
  if (solution_.size() > 0) {
    ceres_problem->SetParameterBlockConstant(solution_[0].pose);
  }
}

// Source moves to target.
double Solver::GetPointCorrespondences(
    const LidarFactor &source_lidar, const LidarFactor &target_lidar,
    double *source_pose, double *target_pose,
    PointCorrespondences *point_correspondences) {
  // Summed differences between point correspondences.
  double difference = 0.0;
  // Affine transformations from the two pose's reference frames.
  Affine2f source_to_world =
      PoseArrayToAffine(&source_pose[2], &source_pose[0]).cast<float>();
  Affine2f target_to_world =
      PoseArrayToAffine(&target_pose[2], &target_pose[0]).cast<float>();
  // Loop over all the points in the source pointcloud,
  // match each point to the closest point in the target pointcloud
  // who's normal is within a certain threshold.
  for (const Vector2f &source_point : source_lidar.pointcloud) {
    // Transform the source point to the target frame.
    Vector2f source_point_transformed =
        target_to_world.inverse() * source_to_world * source_point;
    // Get the closest points within the threshold.
    // For now we assume that a match is within 1/6 of the threshold.
    KDNodeValue<float, 2> closest_target;
    vector<KDNodeValue<float, 2>> neighbors;
    target_lidar.pointcloud_tree->FindNeighborPoints(
        source_point_transformed, SolverConfig::CONFIG_outlier_threshold / 6.0,
        &neighbors);
    // Get the current source point's normal.
    KDNodeValue<float, 2> source_point_with_normal;
    float found_dist = source_lidar.pointcloud_tree->FindNearestPoint(
        source_point, 0.1, &source_point_with_normal);
    CHECK_EQ(found_dist, 0.0) << "Source point is not in KD Tree!\n";
    float dist = SolverConfig::CONFIG_outlier_threshold;
    // Sort the target points by distance from the source point in the
    // target frame.
    std::sort(neighbors.begin(), neighbors.end(),
              [&source_point_transformed](KDNodeValue<float, 2> point_1,
                                          KDNodeValue<float, 2> point_2) {
                return (source_point_transformed - point_1.point).norm() <
                       (source_point_transformed - point_2.point).norm();
              });
    // For all target points, starting with the closest
    // see if any of them have a close enough normal to be
    // considered a match.
    for (KDNodeValue<float, 2> current_target : neighbors) {
      if (NormalsSimilar(current_target.normal, source_point_with_normal.normal,
                         cos(math_util::DegToRad(20.0)))) {
        closest_target = current_target;
        dist = (source_point_transformed - current_target.point).norm();
        break;
      }
    }
    // If we didn't find any matches in the first 1/6 of the threshold,
    // try all target points within the full threshold.
    if (dist >= SolverConfig::CONFIG_outlier_threshold) {
      // Re-find all the closest targets.
      neighbors.clear();
      target_lidar.pointcloud_tree->FindNeighborPoints(
          source_point_transformed, SolverConfig::CONFIG_outlier_threshold,
          &neighbors);
      // Sort them again, based on distance from the source point in the
      // target frame.
      std::sort(neighbors.begin(), neighbors.end(),
                [&source_point_transformed](KDNodeValue<float, 2> point_1,
                                            KDNodeValue<float, 2> point_2) {
                  return (source_point_transformed - point_1.point).norm() <
                         (source_point_transformed - point_2.point).norm();
                });
      // Cut out the first 1/6 threshold that we already checked.
      vector<KDNodeValue<float, 2>> unchecked_neighbors(
          neighbors.begin() + (SolverConfig::CONFIG_outlier_threshold / 6),
          neighbors.end());
      // See if any of these points have a normal within our threshold.
      for (KDNodeValue<float, 2> current_target : unchecked_neighbors) {
        if (NormalsSimilar(current_target.normal,
                           source_point_with_normal.normal,
                           cos(math_util::DegToRad(20.0)))) {
          closest_target = current_target;
          dist = (source_point_transformed - current_target.point).norm();
          break;
        }
      }
      // If no target point was found to correspond to our source point then
      // don't match this source point to anything.
      if (dist >= SolverConfig::CONFIG_outlier_threshold) {
        difference += dist;
        continue;
      }
    }

    // Add the distance between the source point and it's matching target
    // point.
    difference += dist;
    // Add to the correspondence for returning.
    Vector2f source_point_modifiable = source_point;
    point_correspondences->source_points.push_back(source_point_modifiable);
    point_correspondences->target_points.push_back(closest_target.point);
    point_correspondences->source_normals.push_back(
        source_point_with_normal.normal);
    point_correspondences->target_normals.push_back(closest_target.normal);
    // Add a line from the matches that we are using.
    // Transform everything to the world frame.
    // This is for the visualization.
    source_point_transformed = target_to_world * source_point_transformed;
    Vector2f closest_point_in_target = target_to_world * closest_target.point;
    Eigen::Vector3f source_3d(source_point_transformed.x(),
                              source_point_transformed.y(), 0.0f);
    Eigen::Vector3f target_3d(closest_point_in_target.x(),
                              closest_point_in_target.y(), 0.0f);
  }
  return difference;
}

OdometryFactor2D Solver::GetDifferenceOdom(const uint64_t node_a,
                                           const uint64_t node_b) {
  double *pose_a = solution_[node_a].pose;
  double *pose_b = solution_[node_b].pose;
  Vector2f translation(pose_b[0] - pose_a[0], pose_b[1] - pose_a[1]);
  float rotation = pose_b[2] - pose_a[2];
  return OdometryFactor2D(node_a, node_b, translation, rotation);
}

OdometryFactor2D Solver::GetDifferenceOdom(const uint64_t node_a,
                                           const uint64_t node_b,
                                           const Vector3f trans) {
  Vector2f translation(trans[0], trans[1]);
  float rotation = trans[2];
  return OdometryFactor2D(node_a, node_b, translation, rotation);
}

vector<OdometryFactor2D> Solver::GetSolvedOdomFactors() {
  CHECK_GT(solution_.size(), 1);
  vector<OdometryFactor2D> factors;
  for (uint64_t index = 1; index < solution_.size(); index++) {
    // Get the change in translation.
    for (uint64_t prev_idx =
             std::max((uint64_t)0,
                      index - SolverConfig::CONFIG_lidar_constraint_amount_max);
         prev_idx < index; prev_idx++) {
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

vector<OdometryFactor2D> Solver::GetSolvedOdomFactorsBetweenNodes(uint64_t a,
                                                                  uint64_t b) {
  CHECK_GT(solution_.size(), b);
  CHECK_GT(b, a);
  vector<OdometryFactor2D> factors;
  for (uint64_t index = a + 1; index <= b; index++) {
    // Get the change in translation.
    uint64_t prev_idx = index - 1;
    Vector2f prev_loc(solution_[prev_idx].pose[0], solution_[prev_idx].pose[1]);
    Vector2f loc(solution_[index].pose[0], solution_[index].pose[1]);

    double rot_change = math_util::AngleDiff(solution_[index].pose[2],
                                             solution_[prev_idx].pose[2]);
    Vector2f trans_change = loc - prev_loc;
    factors.emplace_back(prev_idx, index, trans_change, rot_change);
  }
  return factors;
}

OdometryFactor2D Solver::GetTotalOdomChange(
    const std::vector<OdometryFactor2D> &factors) {
  Vector2f init_trans(0, 0);
  OdometryFactor2D factor(0, factors.size(), init_trans, 0);
  for (size_t factor_idx = 0; factor_idx < factors.size(); factor_idx++) {
    OdometryFactor2D curr_factor = factors[factor_idx];
    factor.translation += curr_factor.translation;
    factor.rotation += curr_factor.rotation;
    factor.rotation = math_util::AngleMod(factor.rotation);
  }
  return factor;
}

/*----------------------------------------------------------------------------*
 *                        HUMAN-IN-THE-LOOP LOOP CLOSURE                      |
 *----------------------------------------------------------------------------*/

vector<LineSegment<float>> LineSegmentsFromHitlMsg(
    const HitlSlamInputMsg &msg) {
  Vector2f start_a(msg.line_a_start.x, msg.line_a_start.y);
  Vector2f end_a(msg.line_a_end.x, msg.line_a_end.y);
  Vector2f start_b(msg.line_b_start.x, msg.line_b_start.y);
  Vector2f end_b(msg.line_b_end.x, msg.line_b_end.y);
  vector<LineSegment<float>> lines;
  lines.emplace_back(start_a, end_a);
  lines.emplace_back(start_b, end_b);
  return lines;
}

HitlLCConstraint Solver::GetRelevantPosesForHITL(
    const HitlSlamInputMsg &hitl_msg) {
  // Linearly go through all poses
  // Go through all points and see if they lie on either of the two lines.
  const vector<LineSegment<float>> lines = LineSegmentsFromHitlMsg(hitl_msg);
  HitlLCConstraint hitl_constraint(lines[0], lines[1]);
  for (size_t node_idx = 0; node_idx < problem_.nodes.size(); node_idx++) {
    vector<Vector2f> points_on_a;
    vector<Vector2f> points_on_b;
    double *pose_ptr = solution_[node_idx].pose;
    Affine2f node_to_world =
        PoseArrayToAffine(&pose_ptr[2], &pose_ptr[0]).cast<float>();
    for (const Vector2f &point :
         problem_.nodes[node_idx].lidar_factor.pointcloud) {
      Vector2f point_transformed = node_to_world * point;
      if (DistanceToLineSegment(point_transformed, lines[0]) <=
          SolverConfig::CONFIG_hitl_line_width) {
        points_on_a.push_back(point);
      } else if (DistanceToLineSegment(point_transformed, lines[1]) <=
                 SolverConfig::CONFIG_hitl_line_width) {
        points_on_b.push_back(point);
      }
    }
    if (points_on_a.size() >=
        static_cast<size_t>(SolverConfig::CONFIG_hitl_pose_point_threshold)) {
      hitl_constraint.line_a_poses.emplace_back(node_idx, points_on_a);
    } else if (points_on_b.size() >=
               static_cast<size_t>(
                   SolverConfig::CONFIG_hitl_pose_point_threshold)) {
      hitl_constraint.line_b_poses.emplace_back(node_idx, points_on_b);
    }
  }
  return hitl_constraint;
}

void Solver::AddHITLResiduals(ceres::Problem *problem) {
  for (HitlLCConstraint &constraint : hitl_constraints_) {
    for (const LCPose &a_pose : constraint.line_a_poses) {
      const vector<Vector2f> &pointcloud_a = a_pose.points_on_feature;
      CHECK_LT(a_pose.node_idx, solution_.size());
      problem->AddResidualBlock(
          PointToLineResidual::create(constraint.line_a, pointcloud_a), NULL,
          solution_[a_pose.node_idx].pose, constraint.chosen_line_pose);
    }
    for (const LCPose &b_pose : constraint.line_b_poses) {
      const vector<Vector2f> &pointcloud_b = b_pose.points_on_feature;
      CHECK_LT(b_pose.node_idx, solution_.size());
      problem->AddResidualBlock(
          PointToLineResidual::create(constraint.line_a, pointcloud_b), NULL,
          solution_[b_pose.node_idx].pose, constraint.chosen_line_pose);
    }
  }
}

void Solver::HitlCallback(const HitlSlamInputMsgConstPtr &hitl_ptr) {
  problem_.odometry_factors = GetSolvedOdomFactors();
  const HitlSlamInputMsg hitl_msg = *hitl_ptr;
  // Get the poses that belong to this input.
  const HitlLCConstraint colinear_constraint =
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
  problem_.odometry_factors = initial_odometry_factors;
  std::cout << "Solving problem with initial odometry constraints..."
            << std::endl;
  SolveSLAM();
  std::cout << "Waiting for Loop Closure input." << std::endl;
}

vector<ResidualDesc> Solver::AddLCResiduals(const uint64_t node_a,
                                            const uint64_t node_b) {
  const uint64_t first_node = std::min(node_a, node_b);
  const uint64_t second_node = std::max(node_a, node_b);
  vector<ResidualDesc> res_desc;
  PointCorrespondences correspondence(solution_[first_node].pose,
                                      solution_[second_node].pose, first_node,
                                      second_node);
  // Get the correspondences between these two poses.
  GetPointCorrespondences(problem_.nodes[first_node].lidar_factor,
                          problem_.nodes[second_node].lidar_factor,
                          solution_[first_node].pose,
                          solution_[second_node].pose, &correspondence);
  // Add the correspondences as constraints in the optimization problem.
  ceres::ResidualBlockId lidar_id;
  lidar_id = ceres_information.problem->AddResidualBlock(
      LIDARPointBlobResidual::create(
          correspondence.source_points, correspondence.target_points,
          correspondence.source_normals, correspondence.target_normals),
      NULL, correspondence.source_pose, correspondence.target_pose);
  res_desc.emplace_back(node_a, node_b, lidar_id);
  // TODO: Odom factor is difference between solution positions as we haven't
  // solved using CSM yet and can't get the actual difference between them.
  ceres::ResidualBlockId odom_id;
  odom_id = ceres_information.problem->AddResidualBlock(
      OdometryResidual::create(GetDifferenceOdom(first_node, second_node),
                               SolverConfig::CONFIG_lc_translation_weight,
                               SolverConfig::CONFIG_lc_rotation_weight),
      NULL, solution_[first_node].pose, solution_[second_node].pose);
  res_desc.emplace_back(node_a, node_b, odom_id);
  return res_desc;
}

/*----------------------------------------------------------------------------*
 *                            MISC. SOLVER CALLBACKS                          |
 *----------------------------------------------------------------------------*/

void Solver::LoadSLAMSolution(const char *poses_path) {
  std::map<double, Vector3f> poses;
  std::ifstream poses_file;
  poses_file.open(poses_path);
  if (poses_file.is_open()) {
    double timestamp;
    float pose_x, pose_y, theta;
    while (poses_file >> timestamp >> pose_x >> pose_y >> theta) {
      poses[timestamp] = Vector3f(pose_x, pose_y, theta);
    }
  }
  poses_file.close();
  std::cout << "Finished loading solution file." << std::endl;
  for (size_t i = 0; i < solution_.size(); i++) {
    std::stringstream ss;
    ss << std::fixed << solution_[i].timestamp;
    double timestamp = std::stod(ss.str());
    if (poses.find(timestamp) != poses.end()) {
      solution_[i].pose[0] = poses[timestamp][0];
      solution_[i].pose[1] = poses[timestamp][1];
      solution_[i].pose[2] = poses[timestamp][2];
    } else {
      printf("Unable to find solution for timestamp %f\n", timestamp);
    }
  }

  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i < 5; i++) {
    vis_callback_->PubVisualization();
    sleep(1);
  }
}

void Solver::WriteCallback(const WriteMsgConstPtr &msg) {
  if (SolverConfig::CONFIG_pose_output_file.compare("") == 0) {
    std::cout << "No output file specified, not writing!" << std::endl;
    return;
  }
  std::cout << "Writing Poses" << std::endl;
  std::ofstream output_file;
  output_file.open(SolverConfig::CONFIG_pose_output_file);
  for (const SLAMNodeSolution2D &sol_node : solution_) {
    output_file << std::fixed << sol_node.timestamp << " " << sol_node.pose[0]
                << " " << sol_node.pose[1] << " " << sol_node.pose[2]
                << std::endl;
  }
  output_file.close();
}

void Solver::Vectorize(const WriteMsgConstPtr &msg) {
  std::cout << "Vectorizing" << std::endl;
  using VectorMaps::LineSegment;
  vector<Vector2f> whole_pointcloud;
  for (const SLAMNode2D &n : problem_.nodes) {
    vector<Vector2f> pc = n.lidar_factor.pointcloud;
    pc = TransformPointcloud(solution_[n.node_idx].pose, pc);
    whole_pointcloud.insert(whole_pointcloud.begin(), pc.begin(), pc.end());
  }

  vector<LineSegment> lines = VectorMaps::ExtractLines(whole_pointcloud);
  // --- Visualize ---
  visualization_msgs::Marker line_mark;
  gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                gui_helpers::Color4f::kWhite, 0.05, 0.00, 0.00,
                                &line_mark);
  ros::Publisher lines_pub =
      n_.advertise<visualization_msgs::Marker>("/debug_lines", 10);
  for (const LineSegment &line : lines) {
    Vector3f line_start(line.start_point.x(), line.start_point.y(), 0.0);
    Vector3f line_end(line.end_point.x(), line.end_point.y(), 0.0);
    gui_helpers::AddLine(line_start, line_end, gui_helpers::Color4f::kWhite,
                         &line_mark);
  }
  std::cout << "Created map: Pointcloud size: " << whole_pointcloud.size()
            << "\tLines size: " << lines.size() << std::endl;

  if (SolverConfig::CONFIG_map_output_file.compare("") != 0) {
    std::cout << "Writing map to file..." << std::endl;
    std::ofstream output_file;
    output_file.open(SolverConfig::CONFIG_map_output_file);
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
