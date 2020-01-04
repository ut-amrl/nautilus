//
// Created by jack on 9/25/19.
//

#include "./solver.h"
#include <algorithm>
#include <thread>
#include <vector>

#include "ceres/ceres.h"
#include "eigen3/Eigen/Geometry"

#include "./kdtree.h"
#include "./math_util.h"
#include "timer.h"
#include "lidar_slam/HitlSlamInputMsg.h"

#define LIDAR_CONSTRAINT_AMOUNT 10
#define OUTLIER_THRESHOLD 0.25
#define HITL_LINE_WIDTH 0.05
#define HITL_POSE_POINT_THRESHOLD 3

using std::vector;
using slam_types::OdometryFactor2D;
using slam_types::LidarFactor;
using ceres::AutoDiffCostFunction;
using Eigen::Matrix2f;
using Eigen::Vector2f;
using Eigen::Rotation2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using Eigen::Affine2f;
using lidar_slam::HitlSlamInputMsgConstPtr;
using lidar_slam::HitlSlamInputMsg;
using slam_types::SLAMNode2D;
using Eigen::Vector3f;
using math_util::NormalsSimilar;

// Line needs an affine.
// Penalize Radial distance as well.
// Be able to override bag file max distance.

struct OdometryResidual {
  template <typename T>
  bool operator() (const T* pose_i,
                   const T* pose_j,
                   T* residual) const {
    // Predicted pose_j = pose_i * odometry.
    // Hence, error = pose_j.inverse() * pose_i * odometry;
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
//    typedef Eigen::Matrix<T, 2, 2> Matrix2T;
    // Extract the rotation matrices.
//    const Matrix2T Ri = Rotation2D<T>(pose_i[2])
//      .toRotationMatrix();
//    const Matrix2T Rj = Rotation2D<T>(pose_j[2])
//      .toRotationMatrix();
    // Extract the translation.
    const Vector2T Ti(pose_i[0], pose_i[1]);
    const Vector2T Tj(pose_j[0], pose_j[1]);
    // The Error in the translation is the difference with the odometry
    // in the direction of the previous pose, then getting rid of the new
    // rotation (transpose = inverse for rotation matrices).
    const Vector2T error_translation = Ti + T_odom.cast<T>() - Tj;
    // Rotation error is very similar to the translation error, except
    // we don't care about the difference in the position.
    const T error_rotation = pose_i[2] + T(R_odom) - pose_j[2];
    // The residuals are weighted according to the parameters set
    // by the user.
    residual[0] = T(translation_weight) * error_translation.x();
    residual[1] = T(translation_weight) * error_translation.y();
    residual[2] = T(rotation_weight) * error_rotation;
    return true;
  }

  OdometryResidual(const OdometryFactor2D& factor,
                   double translation_weight,
                   double rotation_weight) :
          translation_weight(translation_weight),
          rotation_weight(rotation_weight),
          R_odom(factor.rotation),
          T_odom(factor.translation) {}

  static AutoDiffCostFunction<OdometryResidual, 3, 3, 3>*
  create(const OdometryFactor2D& factor,
         double translation_weight,
         double rotation_weight) {
    OdometryResidual* residual = new OdometryResidual(factor,
                                                      translation_weight,
                                                      rotation_weight);
    return new AutoDiffCostFunction<OdometryResidual, 3, 3, 3>(residual);
  }

  double translation_weight;
  double rotation_weight;
  const float R_odom;
  const Vector2f T_odom;
};

struct LIDARPointBlobResidual {

  // TODO: Add the source normals penalization as well.
  // Would cause there to be two normals.
  template <typename T>
  bool operator() (const T* source_pose,
                   const T* target_pose,
                   T* residuals) const {
    typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
    const Affine2T source_to_world =
      PoseArrayToAffine(&source_pose[2], &source_pose[0]);
    const Affine2T world_to_target =
      PoseArrayToAffine(&target_pose[2], &target_pose[0]).inverse();
    const Affine2T source_to_target = world_to_target * source_to_world;
    #pragma omp parallel for default(none) shared(residuals)
    for (size_t index = 0; index < source_points.size(); index++) {
      Vector2T source_pointT = source_points[index].cast<T>();
      Vector2T target_pointT = target_points[index].cast<T>();
      // Transform source_point into the frame of target_point
      source_pointT = source_to_target * source_pointT;
      T target_normal_result =
        target_normals[index].cast<T>().dot(source_pointT - target_pointT);
      T source_normal_result =
        source_normals[index].cast<T>().dot(target_pointT - source_pointT);
      residuals[index * 2] = target_normal_result;
      residuals[index * 2 + 1] = source_normal_result;
    }
    return true;
  }

  LIDARPointBlobResidual(vector<Vector2f>& source_points,
                         vector<Vector2f>& target_points,
                         vector<Vector2f>& source_normals,
                         vector<Vector2f>& target_normals) :
                         source_points(source_points),
                         target_points(target_points),
                         source_normals(source_normals),
                         target_normals(target_normals) {
    CHECK_EQ(source_points.size(), target_points.size());
    CHECK_EQ(target_points.size(), target_normals.size());
    CHECK_EQ(source_normals.size(), target_normals.size());
  }

  static AutoDiffCostFunction<LIDARPointBlobResidual, ceres::DYNAMIC, 3, 3>*
  create(vector<Vector2f>& source_points,
         vector<Vector2f>& target_points,
         vector<Vector2f>& source_normals,
         vector<Vector2f>& target_normals) {
    LIDARPointBlobResidual *residual =
      new LIDARPointBlobResidual(source_points,
                                 target_points,
                                 source_normals,
                                 target_normals);
    return new AutoDiffCostFunction<LIDARPointBlobResidual,
                                    ceres::DYNAMIC,
                                    3, 3>
                                    (residual, source_points.size() * 2);
  }

  const vector<Vector2f> source_points;
  const vector<Vector2f> target_points;
  const vector<Vector2f> source_normals;
  const vector<Vector2f> target_normals;
};

void Solver::AddOdomFactors(ceres::Problem* ceres_problem,
                            vector<OdometryFactor2D> factors,
                            double trans_weight,
                            double rot_weight) {
  for (const OdometryFactor2D& odom_factor : factors) {
    CHECK_LT(odom_factor.pose_i, odom_factor.pose_j);
    CHECK_GT(solution_.size(), odom_factor.pose_i);
    CHECK_GT(solution_.size(), odom_factor.pose_j);
    ceres_problem->AddResidualBlock(
      OdometryResidual::create(odom_factor, trans_weight, rot_weight),
      NULL,
      solution_[odom_factor.pose_i].pose,
      solution_[odom_factor.pose_j].pose);
  }
  if (solution_.size() > 0) {
    ceres_problem->SetParameterBlockConstant(solution_[0].pose);
  }
}

// Source moves to target.
double
Solver::GetPointCorrespondences(const SLAMProblem2D& problem,
                                vector<SLAMNodeSolution2D>* solution_ptr,
                                PointCorrespondences* point_correspondences,
                                size_t source_node_index,
                                size_t target_node_index) {
  // Summed differences between point correspondences.
  double difference = 0.0;
  vector<SLAMNodeSolution2D> &solution = *solution_ptr;
  SLAMNodeSolution2D &source_solution = solution[source_node_index];
  SLAMNodeSolution2D &target_solution = solution[target_node_index];
  LidarFactor source_lidar =
          problem.nodes[source_node_index].lidar_factor;
  LidarFactor target_lidar =
          problem.nodes[target_node_index].lidar_factor;
  // Affine transformations from the two pose's reference frames.
  Affine2f source_to_world =
          PoseArrayToAffine(&source_solution.pose[2],
                            &source_solution.pose[0]).cast<float>();
  Affine2f target_to_world =
          PoseArrayToAffine(&target_solution.pose[2],
                            &target_solution.pose[0]).cast<float>();
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
    target_lidar.pointcloud_tree->FindNeighborPoints(source_point_transformed,
                                                     OUTLIER_THRESHOLD / 6.0,
                                                     &neighbors);
    // Get the current source point's normal.
    KDNodeValue<float, 2> source_point_with_normal;
    float found_dist =
      source_lidar.pointcloud_tree->FindNearestPoint(source_point,
                                                     0.1,
                                                     &source_point_with_normal);
    CHECK_EQ(found_dist, 0.0) << "Source point is not in KD Tree!\n";
    float dist = OUTLIER_THRESHOLD;
    // Sort the target points by distance from the source point in the
    // target frame.
    std::sort(neighbors.begin(),
              neighbors.end(),
              [&source_point_transformed](KDNodeValue<float, 2> point_1,
                                          KDNodeValue<float, 2> point_2) {
                  return (source_point_transformed - point_1.point).norm() <
                         (source_point_transformed - point_2.point).norm();
              });
    // For all target points, starting with the closest
    // see if any of them have a close enough normal to be 
    // considered a match.
    for (KDNodeValue<float, 2> current_target : neighbors) {
      if (NormalsSimilar(current_target.normal,
                         source_point_with_normal.normal,
                         cos(math_util::DegToRad(20.0)))) {
        closest_target = current_target;
        dist = (source_point_transformed - current_target.point).norm();
        break;
      }
    }
    // If we didn't find any matches in the first 1/6 of the threshold,
    // try all target points within the full threshold.
    if (dist >= OUTLIER_THRESHOLD) {
      // Re-find all the closest targets.
      neighbors.clear();
      target_lidar.pointcloud_tree->FindNeighborPoints(source_point_transformed,
                                                       OUTLIER_THRESHOLD,
                                                       &neighbors);
      // Sort them again, based on distance from the source point in the
      // target frame.
      std::sort(neighbors.begin(),
                neighbors.end(),
                [&source_point_transformed](KDNodeValue<float, 2> point_1,
                                            KDNodeValue<float, 2> point_2) {
                    return (source_point_transformed - point_1.point).norm() <
                           (source_point_transformed - point_2.point).norm();
                });
      // Cut out the first 1/6 threshold that we already checked.
      vector<KDNodeValue<float, 2>>
              unchecked_neighbors(neighbors.begin() + (OUTLIER_THRESHOLD / 6),
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
      if (dist >= OUTLIER_THRESHOLD) {
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
    point_correspondences->
            source_normals.push_back(source_point_with_normal.normal);
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

double Solver::AddLidarResidualsForLC(ceres::Problem& problem) {
  double difference = 0.0;
  for (const LCConstraint constraint : loop_closure_constraints_) {
    for (const LCPose& a_pose : constraint.line_a_poses) {
      for (const LCPose& b_pose : constraint.line_b_poses) {
        CHECK_LT(a_pose.node_idx, solution_.size());
        CHECK_LT(b_pose.node_idx, solution_.size());
        CHECK_LT(a_pose.node_idx, problem_.nodes.size());
        CHECK_LT(b_pose.node_idx, problem_.nodes.size());
        PointCorrespondences correspondence(solution_[a_pose.node_idx].pose,
                                            solution_[b_pose.node_idx].pose,
                                            a_pose.node_idx,
                                            b_pose.node_idx);
        difference += GetPointCorrespondences(problem_,
                                              &solution_,
                                              &correspondence,
                                              a_pose.node_idx,
                                              b_pose.node_idx);
        difference /=
                problem_.nodes[a_pose.node_idx].lidar_factor.pointcloud.size();
        // Add the correspondences as constraints in the optimization problem.
        problem.AddResidualBlock(
                LIDARPointBlobResidual::create(correspondence.source_points,
                                               correspondence.target_points,
                                               correspondence.source_normals,
                                               correspondence.target_normals),
                NULL,
                correspondence.source_pose,
                correspondence.target_pose);
      }
    }
  }
  return difference;
}

vector<SLAMNodeSolution2D>
Solver::SolveSLAM() {
  // Setup ceres for evaluation of the problem.
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  vis_callback_ =
    std::unique_ptr<VisualizationCallback>(
      new VisualizationCallback(problem_, &solution_, n_));
  options.callbacks.push_back(vis_callback_.get());
  double difference = 0;
  double last_difference = 0;
  // While our solution moves more than the stopping_accuracy,
  // continue to optimize.
  for (int64_t window_size = 1;
       window_size <= LIDAR_CONSTRAINT_AMOUNT;
       window_size++) {
    LOG(INFO) << "Using window size: " << window_size << std::endl;
    do {
      vis_callback_->ClearNormals();
      last_difference = difference;
      difference = 0;
      ceres::Problem ceres_problem;
      // Add all the odometry constraints between our poses.
      AddOdomFactors(&ceres_problem,
                     problem_.odometry_factors,
                     translation_weight_,
                     rotation_weight_);
      // For every SLAM node we want to optimize it against the past
      // LIDAR_CONSTRAINT_AMOUNT nodes.
      for (size_t node_i_index = 0;
           node_i_index < problem_.nodes.size();
           node_i_index++) {
        for (size_t node_j_index =
                std::max((int64_t) (node_i_index) - window_size,
                         0l);
             node_j_index < node_i_index;
             node_j_index++) {
          PointCorrespondences correspondence(solution_[node_i_index].pose,
                                              solution_[node_j_index].pose,
                                              node_i_index,
                                              node_j_index);
          // Get the correspondences between these two poses.
          difference += GetPointCorrespondences(problem_,
                                                &solution_,
                                                &correspondence,
                                                node_i_index,
                                                node_j_index);
          vis_callback_->UpdateLastCorrespondence(correspondence);
          difference /=
            problem_.nodes[node_j_index].lidar_factor.pointcloud.size();
          // Add the correspondences as constraints in the optimization problem.
          ceres_problem.AddResidualBlock(
                  LIDARPointBlobResidual::create(correspondence.source_points,
                                                 correspondence.target_points,
                                                 correspondence.source_normals,
                                                 correspondence.target_normals),
                  NULL,
                  correspondence.source_pose,
                  correspondence.target_pose);
        }
      }
      difference += AddLidarResidualsForLC(ceres_problem);
      AddCollinearResiduals(&ceres_problem);
      ceres::Solve(options, &ceres_problem, &summary);
    } while (abs(difference - last_difference) > stopping_accuracy_);
  }
  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i < 5; i++) {
    vis_callback_->PubVisualization();
    sleep(1);
  }
  return solution_;
}

Solver::Solver(double translation_weight,
               double rotation_weight,
               double lc_translation_weight,
               double lc_rotation_weight,
               double stopping_accuracy,
               SLAMProblem2D& problem,
               ros::NodeHandle& n) :
               translation_weight_(translation_weight),
               rotation_weight_(rotation_weight),
               lc_translation_weight_(lc_translation_weight),
               lc_rotation_weight_(lc_rotation_weight),
               stopping_accuracy_(stopping_accuracy),
               problem_(problem),
               n_(n) {
  // Copy all the data to a list that we are going to modify as we optimize.
  for (size_t i = 0; i < problem.nodes.size(); i++) {
    // Make sure that we marked all the data correctly earlier.
    CHECK_EQ(i, problem.nodes[i].node_idx);
    SLAMNodeSolution2D sol_node(problem.nodes[i]);
    solution_.push_back(sol_node);
  }
  CHECK_EQ(solution_.size(), problem.nodes.size());
}

/*
 * Methods relevant to HITL loop closure.
 */

struct PointToLineResidual {
    template <typename T>
    bool operator() (const T* pose, const T* line_pose, T* residuals) const {
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
      const Affine2T pose_to_world =
        PoseArrayToAffine(&pose[2], &pose[0]);
      const Affine2T line_to_world =
        PoseArrayToAffine(&line_pose[2], &line_pose[0]);
      Vector2T line_start = line_to_world * line_segment_.start.cast<T>();
      Vector2T line_end = line_to_world * line_segment_.end.cast<T>();
      const LineSegment<T> TransformedLineSegment(line_start, line_end);
      #pragma omp parallel for default(none) shared(residuals)
      for (size_t index = 0; index < points_.size(); index++) {
        Vector2T pointT = points_[index].cast<T>();
        // Transform source_point into the frame of the line
        pointT = pose_to_world * pointT;
        T dist_along_normal =
          DistanceToLineSegment(pointT, TransformedLineSegment);
        residuals[index] = dist_along_normal;
      }
      return true;
    }

    PointToLineResidual(const LineSegment<float>& line_segment,
                        const vector<Vector2f> points) :
                        line_segment_(line_segment),
                        points_(points) {}

    static AutoDiffCostFunction<PointToLineResidual, ceres::DYNAMIC, 3, 3>*
    create(const LineSegment<float>& line_segment, const vector<Vector2f> points) {
      PointToLineResidual* res = new PointToLineResidual(line_segment, points);
      return new AutoDiffCostFunction<PointToLineResidual,
                                      ceres::DYNAMIC,
                                      3, 3>(res, points.size());
    }

    const LineSegment<float> line_segment_;
    const vector<Vector2f> points_;
};

vector<LineSegment<float>>
LineSegmentsFromHitlMsg(const HitlSlamInputMsg& msg) {
  Vector2f start_a(msg.line_a_start.x, msg.line_a_start.y);
  Vector2f end_a(msg.line_a_end.x, msg.line_a_end.y);
  Vector2f start_b(msg.line_b_start.x, msg.line_b_start.y);
  Vector2f end_b(msg.line_b_end.x, msg.line_b_end.y);
  vector<LineSegment<float>> lines;
  lines.emplace_back(start_a, end_a);
  lines.emplace_back(start_b, end_b);
  return lines;
}

LCConstraint
Solver::GetRelevantPosesForHITL(const HitlSlamInputMsg& hitl_msg) {
  // Linearly go through all poses
  // Go through all points and see if they lie on either of the two lines.
  const vector<LineSegment<float>> lines = LineSegmentsFromHitlMsg(hitl_msg);
  LCConstraint hitl_constraint(lines[0], lines[1]);
  for (size_t node_idx = 0; node_idx < problem_.nodes.size(); node_idx++) {
    vector<Vector2f> points_on_a;
    vector<Vector2f> points_on_b;
    double *pose_ptr = solution_[node_idx].pose;
    Affine2f node_to_world =
            PoseArrayToAffine(&pose_ptr[2], &pose_ptr[0]).cast<float>();
    for (const Vector2f& point : problem_.nodes[node_idx].lidar_factor.pointcloud) {
      Vector2f point_transformed = node_to_world * point;
      if (DistanceToLineSegment(point_transformed, lines[0]) <=
          HITL_LINE_WIDTH) {
        points_on_a.push_back(point);
      } else if (DistanceToLineSegment(point_transformed, lines[1]) <=
                 HITL_LINE_WIDTH) {
        points_on_b.push_back(point);
      }
    }
    if (points_on_a.size() >= HITL_POSE_POINT_THRESHOLD) {
      hitl_constraint.line_a_poses.emplace_back(node_idx, points_on_a);
    } else if (points_on_b.size() >= HITL_POSE_POINT_THRESHOLD) {
      hitl_constraint.line_b_poses.emplace_back(node_idx, points_on_b);
    }
  }
  return hitl_constraint;
}

void Solver::AddColinearConstraints(const LCConstraint& constraint) {
  if (constraint.line_a_poses.size() == 0 ||
      constraint.line_b_poses.size() == 0) {
    return;
  }
  loop_closure_constraints_.push_back(constraint);
  vis_callback_->AddConstraint(constraint);
}

void Solver::AddCollinearResiduals(ceres::Problem* problem) {
  for (LCConstraint& constraint : loop_closure_constraints_) {
    for (const LCPose &a_pose : constraint.line_a_poses) {
      const vector<Vector2f> &pointcloud_a = a_pose.points_on_feature;
      CHECK_LT(a_pose.node_idx, solution_.size());
      problem->AddResidualBlock(
              PointToLineResidual::create(constraint.line_a, pointcloud_a),
              NULL,
              solution_[a_pose.node_idx].pose,
              constraint.chosen_line_pose);
    }
    for (const LCPose &b_pose : constraint.line_b_poses) {
      const vector<Vector2f> &pointcloud_b = b_pose.points_on_feature;
      CHECK_LT(b_pose.node_idx, solution_.size());
      problem->AddResidualBlock(
              PointToLineResidual::create(constraint.line_a, pointcloud_b),
              NULL,
              solution_[b_pose.node_idx].pose,
              constraint.chosen_line_pose);
    }
  }
}

vector<OdometryFactor2D> Solver::GetSolvedOdomFactors() {
  CHECK_GT(solution_.size(), 1);
  vector<OdometryFactor2D> factors;
  for (uint64_t index = 1; index < solution_.size(); index++) {
    // Get the change in translation.
    Vector2f prev_loc(solution_[index - 1].pose[0],
                      solution_[index - 1].pose[1]);
    Vector2f loc(solution_[index].pose[0], solution_[index].pose[1]);
    double rot_change = solution_[index].pose[2] - solution_[index - 1].pose[2];
    Vector2f trans_change = loc - prev_loc;
    factors.emplace_back(index - 1, index, trans_change, rot_change);
  }
  return factors;
}

void Solver::SolveForLC() {
  // Create a new problem with colinear residuals between these nodes and a
  // small weighted odometry residuals between all poses and between these ones.
  ceres::Problem problem;
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.num_threads =
    static_cast<int>(std::thread::hardware_concurrency());
  //options.callbacks.push_back(vis_callback_.get());
  //options.update_state_every_iteration = true;
  vector<OdometryFactor2D> local_factors = GetSolvedOdomFactors();
  AddOdomFactors(&problem,
                 local_factors,
                 lc_translation_weight_,
                 lc_rotation_weight_);
  AddCollinearResiduals(&problem);
  ceres::Solve(options, &problem, &summary);
  vis_callback_->PubVisualization();
  std::cout << summary.FullReport() << std::endl;
}

void Solver::HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr) {
  const HitlSlamInputMsg hitl_msg = *hitl_ptr;
  // Get the poses that belong to this input.
  const LCConstraint collinear_constraint = GetRelevantPosesForHITL(hitl_msg);
  std::cout << "Found " << collinear_constraint.line_a_poses.size()
    << " poses for the first line." << std::endl;
  std::cout << "Found " << collinear_constraint.line_b_poses.size()
            << " poses for the second line." << std::endl;
  AddColinearConstraints(collinear_constraint);
  SolveForLC();
  // Resolve the initial problem with extra pointcloud residuals between these
  // loop closed points.
  SolveSLAM();
  std::cout << "Waiting for Loop Closure input." << std::endl;
}
