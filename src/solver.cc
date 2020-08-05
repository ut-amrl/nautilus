// Created by jack on 9/25/19.
//

#include <algorithm>
#include <fstream>
#include <thread>
#include <vector>

#include <visualization_msgs/Marker.h>
#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "eigen3/Eigen/Sparse"
#include "eigen3/Eigen/SparseQR"
#include <unsupported/Eigen/MatrixFunctions>

#include "./kdtree.h"
#include "./line_extraction.h"
#include "./math_util.h"
#include "nautilus/HitlSlamInputMsg.h"
#include "nautilus/WriteMsg.h"
#include "laser_scan_matcher/MatchLaserScans.h"
#include "local_uncertainty_estimator/EstimateLocalUncertainty.h"
#include "timer.h"
#include <fstream>

#include "./solver.h"
#include "./cimg_debug.h"

#define DEBUG true

using ceres::AutoDiffCostFunction;
using Eigen::Affine2f;
using Eigen::Matrix2f;
using Eigen::Rotation2D;
using Eigen::Vector2f;
using Eigen::Vector3f;
using nautilus::HitlSlamInputMsg;
using nautilus::HitlSlamInputMsgConstPtr;
using nautilus::WriteMsgConstPtr;
using math_util::NormalsSimilar;
using laser_scan_matcher::MatchLaserScans;
using local_uncertainty_estimator::EstimateLocalUncertainty;
using slam_types::LidarFactor;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using std::vector;

/*----------------------------------------------------------------------------*
 *                             CERES RESIDUALS                                |
 *----------------------------------------------------------------------------*/

struct OdometryResidual {
  template <typename T>
  bool operator()(const T* pose_i, const T* pose_j, T* residual) const {
    // Predicted pose_j = pose_i * odometry.
    // Hence, error = pose_j.inverse() * pose_i * odometry;
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
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

  OdometryResidual(const OdometryFactor2D& factor, double translation_weight,
                   double rotation_weight)
      : translation_weight(translation_weight),
        rotation_weight(rotation_weight),
        R_odom(factor.rotation),
        T_odom(factor.translation) {}

  static AutoDiffCostFunction<OdometryResidual, 3, 3, 3>* create(
      const OdometryFactor2D& factor, double translation_weight,
      double rotation_weight) {
    OdometryResidual* residual =
        new OdometryResidual(factor, translation_weight, rotation_weight);
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
  bool operator()(const T* source_pose, const T* target_pose,
                  T* residuals) const {
    typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
    const Affine2T source_to_world =
        PoseArrayToAffine(&source_pose[2], &source_pose[0]);
    const Affine2T world_to_target =
        PoseArrayToAffine(&target_pose[2], &target_pose[0]).inverse();
    const Affine2T source_to_target = world_to_target * source_to_world;
    #pragma omp parallel for shared(residuals)
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
                         vector<Vector2f>& target_normals)
      : source_points(source_points),
        target_points(target_points),
        source_normals(source_normals),
        target_normals(target_normals) {
    CHECK_EQ(source_points.size(), target_points.size());
    CHECK_EQ(target_points.size(), target_normals.size());
    CHECK_EQ(source_normals.size(), target_normals.size());
  }

  static AutoDiffCostFunction<LIDARPointBlobResidual, ceres::DYNAMIC, 3, 3>*
  create(vector<Vector2f>& source_points, vector<Vector2f>& target_points,
         vector<Vector2f>& source_normals, vector<Vector2f>& target_normals) {
    LIDARPointBlobResidual* residual = new LIDARPointBlobResidual(
        source_points, target_points, source_normals, target_normals);
    return new AutoDiffCostFunction<LIDARPointBlobResidual, ceres::DYNAMIC, 3,
                                    3>(residual, source_points.size() * 2);
  }

  const vector<Vector2f> source_points;
  const vector<Vector2f> target_points;
  const vector<Vector2f> source_normals;
  const vector<Vector2f> target_normals;
};

struct PointToLineResidual {
  template <typename T>
  bool operator()(const T* pose, const T* line_pose, T* residuals) const {
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
    typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
    const Affine2T pose_to_world = PoseArrayToAffine(&pose[2], &pose[0]);
    const Affine2T line_to_world =
        PoseArrayToAffine(&line_pose[2], &line_pose[0]);
    Vector2T line_start = line_to_world * line_segment_.start.cast<T>();
    Vector2T line_end = line_to_world * line_segment_.end.cast<T>();
    const LineSegment<T> TransformedLineSegment(line_start, line_end);
    #pragma omp parallel for shared(residuals)
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
                      const vector<Vector2f> points)
      : line_segment_(line_segment), points_(points) {}

  static AutoDiffCostFunction<PointToLineResidual, ceres::DYNAMIC, 3, 3>*
  create(const LineSegment<float>& line_segment,
         const vector<Vector2f> points) {
    PointToLineResidual* res = new PointToLineResidual(line_segment, points);
    return new AutoDiffCostFunction<PointToLineResidual, ceres::DYNAMIC, 3, 3>(
        res, points.size());
  }

  const LineSegment<float> line_segment_;
  const vector<Vector2f> points_;
};

/*----------------------------------------------------------------------------*
 *                             PROBLEM BUILDING                               |
 *----------------------------------------------------------------------------*/

/* Solver takes in a ros node handle to be used for sending out debugging
 * information*/
// TODO: Upped the Scanmatcher resolution to 0.01 for ChiSquare.
Solver::Solver(ros::NodeHandle& n) : n_(n), scan_matcher(30, 2, 0.3, 0.01) {
  matcher_client =
      n_.serviceClient<MatchLaserScans>("match_laser_scans");
  local_uncertainty_client =
      n_.serviceClient<EstimateLocalUncertainty>("estimate_local_uncertainty");
  vis_callback_ = std::unique_ptr<VisualizationCallback>(
      new VisualizationCallback(keyframes, n_));
}

void Solver::AddSLAMNodeOdom(SLAMNode2D& node,
                             OdometryFactor2D& odom_factor_to_node) {
  CHECK_EQ(node.node_idx, odom_factor_to_node.pose_j);
  problem_.nodes.push_back(node);
  problem_.odometry_factors.push_back(odom_factor_to_node);
  initial_odometry_factors.push_back(odom_factor_to_node);
  SLAMNodeSolution2D sol_node(node);
  solution_.push_back(sol_node);
  vis_callback_->UpdateProblemAndSolution(node, &solution_,
                                          odom_factor_to_node);
}

void Solver::AddSlamNode(SLAMNode2D& node) {
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
  // options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  // options.callbacks.push_back(vis_callback_.get());
  // for (int64_t i = 0; i < 20; i++) {
  //   vis_callback_->ClearNormals();
  //   ceres_information.ResetProblem();
  //   // Add all the odometry constraints between our poses.
  //   AddOdomFactors(ceres_information.problem.get(), GetSolvedOdomFactors(),
  //                   config_.CONFIG_translation_weight,
  //                   config_.CONFIG_rotation_weight);
  //   AddResidualsForAutoLC(ceres_information.problem.get(), false);
  //   ceres::Solve(options, ceres_information.problem.get(), &summary);
  // }
  
  // Apply Loop closures to current solution
  // This is COP SLAM
  for(auto constraint : auto_lc_constraints_) {
    uint64_t i = constraint.node_a->node_idx;
    uint64_t j = constraint.node_b->node_idx;
    SLAMNodeSolution2D sol_i = solution_[constraint.node_a->node_idx];
    SLAMNodeSolution2D sol_j = solution_[constraint.node_b->node_idx];
    Eigen::Affine2d A_Mi = Eigen::Translation2d(sol_i.pose[0], sol_i.pose[1]) * Eigen::Rotation2Dd(sol_i.pose[2]);
    Eigen::Affine2d A_Mj = Eigen::Translation2d(sol_j.pose[0], sol_j.pose[1]) * Eigen::Rotation2Dd(sol_j.pose[2]);
    Eigen::Affine2d A_Mj_star = A_Mi * Eigen::Affine2d(Eigen::Translation2d(constraint.relative_transformation[0], constraint.relative_transformation[1]) * Eigen::Rotation2Dd(constraint.relative_transformation[2]));

    std::cout << "Mi" << std::endl;
    std::cout << A_Mi.matrix() << std::endl;
    std::cout << "Mj" << std::endl;
    std::cout << A_Mj.matrix() << std::endl;
    std::cout << "Mj_star" << std::endl;
    std::cout << A_Mj_star.matrix() << std::endl;

    // std::vector<OdometryFactor2D> factors = GetSolvedOdomFactorsBetweenNodes(i, j);
    // OdometryFactor2D chainFactor = GetTotalOdomChange(factors);
    // Eigen::Affine2d A_chain = Eigen::Translation2f(chainFactor.translation) * Eigen::Rotation2Df(chainFactor.rotation);

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
    for(uint64_t k = 1; k < N; k++) {
      Eigen::Matrix3d poseUpdateMat = DeltaA.matrix().pow((double)k / N);
      Eigen::Affine2d poseUpdate(poseUpdateMat);
      solution_[i+k].pose[0] += poseUpdate.translation().x();
      solution_[i+k].pose[1] += poseUpdate.translation().y();
      solution_[i+k].pose[2] += Eigen::Rotation2Dd(poseUpdate.rotation()).angle();
    }

    // Update all subsequent poses
    for(uint64_t k = 0; k < solution_.size() - j; k++) {
      solution_[j+k].pose[0] += DeltaA.translation().x();
      solution_[j+k].pose[1] += DeltaA.translation().y();
      solution_[j+k].pose[2] += Eigen::Rotation2Dd(DeltaA.rotation()).angle();
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
        alpha_1 = std::accumulate(weights.begin(), weights.begin() + k - 1, 0.0);
        alpha_2 = std::accumulate(weights.begin(), weights.begin() + k, 0.0);
      } else {
        alpha_1 = 0;
        alpha_2 = weights[0];
      }

      Affine2f J_1 = Eigen::Affine2f(((alpha_1 * jarg_log).exp()).matrix());
      Affine2f J_2 = Eigen::Affine2f(((alpha_2 * jarg_log).exp()).matrix());
      // Compute local update (8)
      Affine2f update_local = J_1.inverse() * J_2;
      // Compute distributed update (14)
      Eigen::Affine2f A_k = Eigen::Translation2f(solution_[i + k].pose[0], solution_[i + k].pose[1]) * Eigen::Rotation2Df(solution_[i + k].pose[2]);
      Affine2f update_distributed = A_k.inverse() * A_Mj_star * update_local * A_Mj_star.inverse() * A_k;

      std::cout << "UPDATE" << update_distributed.matrix() << std::endl;

      solution_[i+k].pose[0] += update_distributed.translation().x();
      solution_[i+k].pose[1] += update_distributed.translation().y();
      solution_[i+k].pose[2] += Eigen::Rotation2Df(update_distributed.rotation()).angle();
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
  // Setup ceres for evaluation of the problem.
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  options.callbacks.push_back(vis_callback_.get());
  double difference = 0;
  double last_difference = 0;
  // While our solution moves more than the stopping_accuracy,
  // continue to optimize.
  for (int64_t window_size = 1;
       window_size <= config_.CONFIG_lidar_constraint_amount; window_size++) {
    LOG(INFO) << "Using window size: " << window_size << std::endl;
    do {
      vis_callback_->ClearNormals();
      last_difference = difference;
      difference = 0;
      ceres_information.ResetProblem();
      // Add all the odometry constraints between our poses.
      AddOdomFactors(ceres_information.problem.get(), problem_.odometry_factors,
                     config_.CONFIG_translation_weight,
                     config_.CONFIG_rotation_weight);
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
              GetPointCorrespondences(problem_.nodes[node_i_index].lidar_factor, problem_.nodes[node_j_index].lidar_factor,
                                      solution_[node_i_index].pose, solution_[node_j_index].pose,
                                      &correspondence);
          problem_mutex.lock();
          vis_callback_->UpdateLastCorrespondence(correspondence);
          problem_mutex.unlock();
          difference /=
              problem_.nodes[node_j_index].lidar_factor.pointcloud.size();
          // Add the correspondences as constraints in the optimization problem.
          problem_mutex.lock();
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
      difference += AddResidualsForAutoLC(ceres_information.problem.get(), true);
      AddHITLResiduals(ceres_information.problem.get());
      ceres::Solve(options, ceres_information.problem.get(), &summary);
    } while (abs(difference - last_difference) >
             config_.CONFIG_stopping_accuracy);
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
                            double trans_weight, double rot_weight) {
  for (const OdometryFactor2D& odom_factor : factors) {
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
double Solver::GetPointCorrespondences(const LidarFactor& source_lidar, const LidarFactor& target_lidar,
  double* source_pose, double* target_pose, PointCorrespondences* point_correspondences) {
  // Summed differences between point correspondences.
  double difference = 0.0;
  // Affine transformations from the two pose's reference frames.
  Affine2f source_to_world =
      PoseArrayToAffine(&source_pose[2], &source_pose[0])
          .cast<float>();
  Affine2f target_to_world =
      PoseArrayToAffine(&target_pose[2], &target_pose[0])
          .cast<float>();
  // Loop over all the points in the source pointcloud,
  // match each point to the closest point in the target pointcloud
  // who's normal is within a certain threshold.
  for (const Vector2f& source_point : source_lidar.pointcloud) {
    // Transform the source point to the target frame.
    Vector2f source_point_transformed =
        target_to_world.inverse() * source_to_world * source_point;
    // Get the closest points within the threshold.
    // For now we assume that a match is within 1/6 of the threshold.
    KDNodeValue<float, 2> closest_target;
    vector<KDNodeValue<float, 2>> neighbors;
    target_lidar.pointcloud_tree->FindNeighborPoints(
        source_point_transformed, config_.CONFIG_outlier_threshold / 6.0,
        &neighbors);
    // Get the current source point's normal.
    KDNodeValue<float, 2> source_point_with_normal;
    float found_dist = source_lidar.pointcloud_tree->FindNearestPoint(
        source_point, 0.1, &source_point_with_normal);
    CHECK_EQ(found_dist, 0.0) << "Source point is not in KD Tree!\n";
    float dist = config_.CONFIG_outlier_threshold;
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
    if (dist >= config_.CONFIG_outlier_threshold) {
      // Re-find all the closest targets.
      neighbors.clear();
      target_lidar.pointcloud_tree->FindNeighborPoints(
          source_point_transformed, config_.CONFIG_outlier_threshold,
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
          neighbors.begin() + (config_.CONFIG_outlier_threshold / 6),
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
      if (dist >= config_.CONFIG_outlier_threshold) {
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
  double* pose_a = solution_[node_a].pose;
  double* pose_b = solution_[node_b].pose;
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
    for(uint64_t prev_idx = std::max((uint64_t)0, index - config_.CONFIG_lidar_constraint_amount); prev_idx < index; prev_idx++) {
      Vector2f prev_loc(solution_[prev_idx].pose[0],
                        solution_[prev_idx].pose[1]);
      Vector2f loc(solution_[index].pose[0], solution_[index].pose[1]);

      double rot_change = solution_[index].pose[2] - solution_[prev_idx].pose[2];
      Vector2f trans_change = loc - prev_loc;
      factors.emplace_back(prev_idx, index, trans_change, rot_change);
    }
  }
  return factors;
}

vector<OdometryFactor2D> Solver::GetSolvedOdomFactorsBetweenNodes(uint64_t a, uint64_t b) {
  CHECK_GT(solution_.size(), b);
  CHECK_GT(b, a);
  vector<OdometryFactor2D> factors;
  for (uint64_t index = a + 1; index <= b; index++) {
    // Get the change in translation.
    uint64_t prev_idx = index - 1;
    Vector2f prev_loc(solution_[prev_idx].pose[0],
                      solution_[prev_idx].pose[1]);
    Vector2f loc(solution_[index].pose[0], solution_[index].pose[1]);

    double rot_change = math_util::AngleDiff(solution_[index].pose[2], solution_[prev_idx].pose[2]);
    Vector2f trans_change = loc - prev_loc;
    factors.emplace_back(prev_idx, index, trans_change, rot_change);
  }
  return factors;
}

OdometryFactor2D Solver::GetTotalOdomChange(const std::vector<OdometryFactor2D>& factors) {
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
 *                        AUTONOMOUS LOOP CLOSURE                             |
 *----------------------------------------------------------------------------*/

AutoLCConstraint Solver::computeAutoLCConstraint(uint64_t node_a_idx, uint64_t node_b_idx) {
  AutoLCConstraint constraint;
  // Compute the transformation necessary to go from node a to node b, in b's reference frame
  std::pair<double, std::pair<Vector2f, float>> trans_prob_pair =
      scan_matcher.GetTransformation(
          problem_.nodes[node_a_idx].lidar_factor.pointcloud,
          problem_.nodes[node_b_idx].lidar_factor.pointcloud,
          solution_[node_a_idx].pose[2],
          solution_[node_b_idx].pose[2],
          math_util::DegToRad(90));
  auto trans = trans_prob_pair.second;

  constraint.node_a = &problem_.nodes[node_a_idx];
  constraint.node_b = &problem_.nodes[node_b_idx];

  constraint.target_pose[0] = solution_[node_b_idx].pose[0];
  constraint.target_pose[1] = solution_[node_b_idx].pose[1];
  constraint.target_pose[2] = solution_[node_b_idx].pose[2];

  // get the relative transformation between the poses in the global reference frame
  Vector2f relativeTranslation = Eigen::Rotation2Df(constraint.target_pose[2]) * trans.first;
  constraint.relative_transformation[0] = -relativeTranslation.x();
  constraint.relative_transformation[1] = -relativeTranslation.y();
  constraint.relative_transformation[2] = -trans.second;

  constraint.source_pose[0] = solution_[node_b_idx].pose[0] - constraint.relative_transformation[0];
  constraint.source_pose[1] = solution_[node_b_idx].pose[1] - constraint.relative_transformation[1];
  constraint.source_pose[2] = solution_[node_b_idx].pose[2] - constraint.relative_transformation[2];

  // Check the point correspondences to make sure the match is good enough
  PointCorrespondences correspondence(constraint.source_pose,
                                      constraint.target_pose,
                                      constraint.node_a->node_idx, constraint.node_b->node_idx);
  double temp_diff =  GetPointCorrespondences(constraint.node_a->lidar_factor,
                                              constraint.node_b->lidar_factor,
                                              constraint.source_pose, constraint.target_pose, &correspondence);

  temp_diff = temp_diff / constraint.node_a->lidar_factor.pointcloud.size();

  constraint.match_ratio = (double)correspondence.source_points.size() / std::min(constraint.node_a->lidar_factor.pointcloud.size(), constraint.node_b->lidar_factor.pointcloud.size());

  return constraint;
}

bool Solver::AddAutoLCConstraint(const AutoLCConstraint& constraint) {
  #if DEBUG
  std::vector<Vector2f> global_a;
  std::vector<Vector2f> global_b;
  Affine2f target_to_world =
    PoseArrayToAffine(&constraint.target_pose[2], &constraint.target_pose[0])
        .cast<float>();
  Affine2f source_to_world =
    PoseArrayToAffine(&constraint.source_pose[2], &constraint.source_pose[0])
        .cast<float>();
  
  for(const Vector2f& pt : problem_.nodes[constraint.node_a->node_idx].lidar_factor.pointcloud) {
    global_a.push_back(source_to_world * pt);
  }
  
  for(const Vector2f& pt : problem_.nodes[constraint.node_b->node_idx].lidar_factor.pointcloud) {
    global_b.push_back(target_to_world * pt);
  }

  // Draw local points
  // WaitForClose({DrawPoints(problem_.nodes[constraint.node_a->node_idx].lidar_factor.pointcloud), DrawPoints(problem_.nodes[constraint.node_b->node_idx].lidar_factor.pointcloud)});
  // Draw global points
  // WaitForClose({GetTable(global_a, 80.0, 0.15), GetTable(global_b, 80.0, 0.15)});
  #endif

  #if DEBUG
  printf("match ratio %f\n", constraint.match_ratio);
  printf("poses (%f %f %f) (%f %f %f)\n", constraint.source_pose[0], constraint.source_pose[1], constraint.source_pose[2], constraint.target_pose[0], constraint.target_pose[1], constraint.target_pose[2]);
  #endif

  printf("Adding Loop Closure constraint %ld %ld. (%f %f %f)\n",
    constraint.node_a->node_idx, 
    constraint.node_b->node_idx,
    constraint.relative_transformation[0],
    constraint.relative_transformation[1],
    constraint.relative_transformation[2]);

  std::cout << "Writing LC Info" << std::endl;
  std::ofstream lc_output_file;
  lc_output_file.open(config_.CONFIG_lc_debug_output_dir + "/lc_matches.txt", std::ios::app);
  lc_output_file << "Loop Closed " << constraint.node_a->node_idx << " " << constraint.node_b->node_idx
                 << ", transformation: " << constraint.relative_transformation.transpose() << std::endl;
  lc_output_file.close();
  // add constraint
  auto_lc_constraints_.push_back(constraint);
  vis_callback_->AddAutoLCConstraint(constraint);
  vis_callback_->PubVisualization();

  #define DEBUG_AUTO_LC

  #ifdef DEBUG_AUTO_LC
  return true;
  #endif

  // Solve the pose graph problem
  std::cout << "Solving pose problem" << std::endl;
  SolvePoseSLAM();
  std::cout << "Solved pose problem" << std::endl;
  sleep(10);
  SolveSLAM();
  return true;
}

double Solver::AddResidualsForAutoLC(ceres::Problem* problem, bool include_lidar) {
  double difference = 0.0;

  for (const AutoLCConstraint&  constraint : auto_lc_constraints_) {
    // add the odometry residual
    ceres::ResidualBlockId odom_id;
    odom_id = ceres_information.problem->AddResidualBlock(
        OdometryResidual::create(GetDifferenceOdom(constraint.node_a->node_idx, constraint.node_b->node_idx, constraint.relative_transformation),
                                config_.CONFIG_lc_translation_weight,
                                config_.CONFIG_lc_rotation_weight),
        NULL, solution_[constraint.node_a->node_idx].pose, solution_[constraint.node_b->node_idx].pose);
    ceres_information.res_descriptors.emplace_back(constraint.node_a->node_idx, constraint.node_b->node_idx, odom_id);

    #if DEBUG
    printf("Poses: (%f %f %f) (%f %f %f) ... trans (%f %f %f)\n",
      solution_[constraint.node_a->node_idx].pose[0],
      solution_[constraint.node_a->node_idx].pose[1],
      solution_[constraint.node_a->node_idx].pose[2],
      solution_[constraint.node_b->node_idx].pose[0],
      solution_[constraint.node_b->node_idx].pose[1],
      solution_[constraint.node_b->node_idx].pose[2],
      constraint.relative_transformation[0],
      constraint.relative_transformation[1],
      constraint.relative_transformation[2]
    );
    #endif
    
    // add the lidar residuals; Here we assume the `solution_` has already been updated with csm-aligned poses
    if (include_lidar) {
      PointCorrespondences correspondence(solution_[constraint.node_a->node_idx].pose,
                                          solution_[constraint.node_b->node_idx].pose,
                                          constraint.node_a->node_idx, constraint.node_b->node_idx);
      double temp_diff = GetPointCorrespondences(
          constraint.node_a->lidar_factor, constraint.node_b->lidar_factor, solution_[constraint.node_a->node_idx].pose, solution_[constraint.node_b->node_idx].pose, &correspondence);
      
      // Add the correspondences as constraints in the optimization problem.
      ceres::ResidualBlockId id = problem->AddResidualBlock(
          LIDARPointBlobResidual::create(
              correspondence.source_points, correspondence.target_points,
              correspondence.source_normals, correspondence.target_normals),
          NULL, correspondence.source_pose, correspondence.target_pose);
      ceres_information.res_descriptors.emplace_back(constraint.node_a->node_idx,
                                                      constraint.node_b->node_idx, id);

      difference +=
          temp_diff /
          problem_.nodes[constraint.node_a->node_idx].lidar_factor.pointcloud.size();
    }
    // TODO: do I need to add residual for node_b -> node_a...just the mirror of the one I just added?
  }

  return difference;
}

std::pair<double, double> Solver::GetLocalUncertainty(const uint64_t node_idx) {
  if (node_idx <
      static_cast<uint64_t>(config_.CONFIG_local_uncertainty_prev_scans)) {
    return std::make_pair(0, 0);
  }
  std::vector<std::vector<Vector2f>> prevScans;
  for (uint64_t idx = node_idx - 1;
       idx > node_idx - static_cast<uint64_t>(
                            config_.CONFIG_local_uncertainty_prev_scans);
       idx--) {
    prevScans.push_back(problem_.nodes[idx].lidar_factor.pointcloud);
  }
  return scan_matcher.GetLocalUncertaintyStats(
      prevScans, problem_.nodes[node_idx].lidar_factor.pointcloud);
}

void Solver::RemoveResiduals(vector<ResidualDesc> descs) {
  for (const ResidualDesc res_desc : descs) {
    ceres_information.problem->RemoveResidualBlock(res_desc.id);
  }
}

void Solver::AddKeyframe(SLAMNode2D& node) {
  node.is_keyframe = true;
  keyframes.emplace_back(node.node_idx);
}

vector<size_t> Solver::GetMatchingKeyframeIndices(size_t keyframe_index) {
  vector<size_t> matches;
  for (size_t i = 0; i < keyframes.size(); i++) {
    if (i == keyframe_index) {
      continue;
    }
    if (SimilarScans(keyframes[i].node_idx,
                     keyframes[keyframe_index].node_idx, 0.95)) {
      matches.push_back(i);
    }
  }
  return matches;
}

void Solver::CheckForLearnedLC(SLAMNode2D& node) {
  // TODO: Used to add first node as keyframe,
  // now its the 2nd node as the first is constant and therefore
  // has 0 covariance with anything else.
  // double img_width = furthest_point(problem_.nodes[node.node_idx].lidar_factor.pointcloud).norm();
  if (keyframes.size() == 0 && problem_.nodes.size() > 1) {
    AddKeyframe(problem_.nodes[1]);
//    SaveImage(config_.CONFIG_lc_debug_output_dir + "/keyframe_1.bmp",
//    GetTable(problem_.nodes[1].lidar_factor.pointcloud,
//    img_width, 0.03));
    return;
  } else if (keyframes.size() == 0) {
    // Keyframes is empty, but we don't have the 2nd node yet.
    // But we don't want to even check the 1st node, because
    // constant.
    return;
  }

  # if DEBUG
  printf("Processing node %ld\n", node.node_idx);
  #endif
  // Step 1: Check if this is a valid keyframe using the ChiSquared test,
  // basically is it different than the last keyframe.
  if (config_.CONFIG_keyframe_chi_squared_test) {
    if (!SimilarScans(keyframes[keyframes.size() - 1].node_idx, node.node_idx,
                      0.95)) {
      #if DEBUG
      printf("Not a keyframe from chi^2\n");
      #endif
      return;
    }
  } else {
    // Weak emulation of chi^2....only add keyframes after enough time has passed or we have moved far enough
    SLAMNode2D& prev_key_node = problem_.nodes[keyframes[keyframes.size() - 1].node_idx];
    double pose_dist = GetDifferenceOdom(node.node_idx, prev_key_node.node_idx).translation.norm();
    if (pose_dist < config_.CONFIG_keyframe_min_odom_distance) {
      #if DEBUG
      printf("Not a keyframe due to lack of pose uncertainty. total distance %f \n", pose_dist);
      #endif     
      return;
    }
  }

  // Step X: Check if this is a valid scan for loop closure by sub sampling from
  // the scans close to it using local invariance.
  if (config_.CONFIG_keyframe_local_uncertainty_filtering) {
    auto uncertainty = GetLocalUncertaintyEstimate(node.node_idx);
    if (uncertainty.first > config_.CONFIG_local_uncertainty_condition_threshold ||
        uncertainty.second > config_.CONFIG_local_uncertainty_scale_threshold)
        {
      #if DEBUG
      printf("Not a keyframe due to lack of local invariance... Computed Uncertainty: %f, %f\n",
              uncertainty.first,
              uncertainty.second);
      #endif
      return;
    }
    // WaitForClose({DrawPoints(problem_.nodes[node.node_idx].lidar_factor.pointcloud)});
  }
  
  std::cout << "Adding Keyframe # " << keyframes.size() << " at node " << node.node_idx << std::endl;
  AddKeyframe(node);

  // SaveImage(config_.CONFIG_lc_debug_output_dir + "/keyframe_" + std::to_string(node.node_idx) + ".bmp",
  //   GetTable(problem_.nodes[node.node_idx].lidar_factor.pointcloud,
  //   img_width, 0.03));

  // Step 4: Compare against all previous keyframes and see if there is a
  // or is similar using Chi^2
  // vector<size_t> matches = GetMatchingKeyframeIndices(keyframes.size() - 1);
  // if (matches.size() == 0) {
  //   #if DEBUG
  //   printf("No match from chi^2\n");
  //   #endif
  //   return;
  // }

  // Step 5: Compare the embeddings and see if there is a match as well.
  // Find the closest embedding in all the matches to our new keyframe.
  LearnedKeyframe new_keyframe = keyframes[keyframes.size() - 1];
  vector<size_t> matches;

  for (size_t match_index = 0; match_index < keyframes.size() - std::min(config_.CONFIG_lc_min_keyframes, (int)keyframes.size()); match_index++) {
  // for (size_t match_index = 0; match_index < keyframes.size() - 1; match_index++) {
    LearnedKeyframe matched_keyframe = keyframes[match_index];
    SLAMNode2D& keyframe_node = problem_.nodes[matched_keyframe.node_idx];
    if ((node.pose.loc - keyframe_node.pose.loc).norm() > config_.CONFIG_lc_base_max_range + config_.CONFIG_lc_max_range_scaling * (node.node_idx - matched_keyframe.node_idx)) {
      // #if DEBUG
      // printf("Too far away, not considering LC between %ld and %ld\n", node.node_idx, matched_keyframe.node_idx);
      // #endif
      continue;
    }

    float match_score = GetMatchScores(node, keyframe_node);

    if (match_score > config_.CONFIG_lc_match_threshold) {
      matches.push_back(match_index);
    }
  }

  if (matches.size() == 0) {
    #if DEBUG
    printf("No matches found, continuing\n");
    #endif
    return;
  }

  std::ofstream lc_output_file;
  lc_output_file.open(config_.CONFIG_lc_debug_output_dir + "/lc_matches.txt", std::ios::app);
  std::vector<AutoLCConstraint> constraints;
  for(auto idx : matches) {
    #if DEBUG
    printf("Found match of pose %lu to %lu\n",
            keyframes[idx].node_idx,
            new_keyframe.node_idx);
    #endif
    lc_output_file << "Matched " << new_keyframe.node_idx << " " << keyframes[idx].node_idx << std::endl;

    if (config_.CONFIG_lc_match_window_size % 2 != 1) {
      std::cerr << "lc_match_window_size must be odd" << std::endl;
      exit(1);
    }

    size_t window_offset = config_.CONFIG_lc_match_window_size / 2;

    size_t start = std::max(new_keyframe.node_idx - window_offset, (size_t)0);
    size_t end = std::min(new_keyframe.node_idx + window_offset, problem_.nodes.size());

    printf("checking from %ld to %ld\n", start, end);

    bool windowMatched = true;
    AutoLCConstraint keyframeConstraint;
    for(auto i = start; i < end; i++) {
      AutoLCConstraint constraint = computeAutoLCConstraint(keyframes[idx].node_idx, i);
      if (i == new_keyframe.node_idx) {
        keyframeConstraint = constraint;
      }
      if (constraint.match_ratio < 0.75) {
        windowMatched = false;
        printf("Failed to match %ld to %ld\n", keyframes[idx].node_idx, i);
        break;
      }
    }

    if (windowMatched) {
      constraints.push_back(keyframeConstraint);
      lc_output_file << "Validated match " << new_keyframe.node_idx << " " << keyframes[idx].node_idx << std::endl;
    }
  }
  lc_output_file.close();

  // #if DEBUG
  // std::vector<WrappedImage> images =
  //   {DrawPoints(problem_.nodes[new_keyframe.node_idx].lidar_factor.pointcloud),
  //   DrawPoints(problem_.nodes[keyframes[closest_index].node_idx].lidar_factor.pointcloud)};
  // WaitForClose(images);

  // double width =
  // furthest_point(problem_.nodes[new_keyframe.node_idx].lidar_factor.pointcloud).norm();
  // SaveImage("LC_1",
  // GetTable(problem_.nodes[new_keyframe.node_idx].lidar_factor.pointcloud,
  // width, 0.03)); width =
  // furthest_point(problem_.nodes[keyframes[closest_index].node_idx].lidar_factor.pointcloud).norm();
  // SaveImage("LC_2",
  // GetTable(problem_.nodes[keyframes[closest_index].node_idx].lidar_factor.pointcloud,
  // width, 0.03));
  // #endif

  size_t best_idx;
  float best_match = -1;
  for(size_t i = 0; i < constraints.size(); i++) {
    if (constraints[i].match_ratio > best_match) {
      best_idx = i;
      best_match = constraints[i].match_ratio;
    }
  }

  // Step 6: Perform loop closure between these poses if there is a LC.
  if (best_match > -1) {
    AddAutoLCConstraint(constraints[best_idx]);
  }
}

/*----------------------------------------------------------------------------*
 *                        HUMAN-IN-THE-LOOP LOOP CLOSURE                      |
 *----------------------------------------------------------------------------*/

vector<LineSegment<float>> LineSegmentsFromHitlMsg(
    const HitlSlamInputMsg& msg) {
  Vector2f start_a(msg.line_a_start.x, msg.line_a_start.y);
  Vector2f end_a(msg.line_a_end.x, msg.line_a_end.y);
  Vector2f start_b(msg.line_b_start.x, msg.line_b_start.y);
  Vector2f end_b(msg.line_b_end.x, msg.line_b_end.y);
  vector<LineSegment<float>> lines;
  lines.emplace_back(start_a, end_a);
  lines.emplace_back(start_b, end_b);
  return lines;
}

HitlLCConstraint Solver::GetRelevantPosesForHITL(const HitlSlamInputMsg& hitl_msg) {
  // Linearly go through all poses
  // Go through all points and see if they lie on either of the two lines.
  const vector<LineSegment<float>> lines = LineSegmentsFromHitlMsg(hitl_msg);
  HitlLCConstraint hitl_constraint(lines[0], lines[1]);
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
  for (HitlLCConstraint& constraint : hitl_constraints_) {
    for (const LCPose& a_pose : constraint.line_a_poses) {
      const vector<Vector2f>& pointcloud_a = a_pose.points_on_feature;
      CHECK_LT(a_pose.node_idx, solution_.size());
      problem->AddResidualBlock(
          PointToLineResidual::create(constraint.line_a, pointcloud_a), NULL,
          solution_[a_pose.node_idx].pose, constraint.chosen_line_pose);
    }
    for (const LCPose& b_pose : constraint.line_b_poses) {
      const vector<Vector2f>& pointcloud_b = b_pose.points_on_feature;
      CHECK_LT(b_pose.node_idx, solution_.size());
      problem->AddResidualBlock(
          PointToLineResidual::create(constraint.line_a, pointcloud_b), NULL,
          solution_[b_pose.node_idx].pose, constraint.chosen_line_pose);
    }
  }
}

void Solver::HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr) {
  problem_.odometry_factors = GetSolvedOdomFactors();
  const HitlSlamInputMsg hitl_msg = *hitl_ptr;
  // Get the poses that belong to this input.
  const HitlLCConstraint colinear_constraint = GetRelevantPosesForHITL(hitl_msg);
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
  //TODO: This is giving worse results.
  problem_.odometry_factors = initial_odometry_factors;
  std::cout << "Solving problem with initial odometry constraints..." << std::endl;
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
  GetPointCorrespondences(problem_.nodes[first_node].lidar_factor, problem_.nodes[second_node].lidar_factor,
                          solution_[first_node].pose, solution_[second_node].pose,
                          &correspondence);
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
                               config_.CONFIG_lc_translation_weight,
                               config_.CONFIG_lc_rotation_weight),
      NULL, solution_[first_node].pose, solution_[second_node].pose);
  res_desc.emplace_back(node_a, node_b, odom_id);
  return res_desc;
}

/*----------------------------------------------------------------------------*
 *                            MISC. SOLVER CALLBACKS                          |
 *----------------------------------------------------------------------------*/

bool Solver::SimilarScans(const uint64_t node_a, const uint64_t node_b,
                          const double certainty) {
  if (node_a == node_b) {
    return true;
  }
  CHECK_LE(certainty, 1.0);
  CHECK_GE(certainty, 0.0);
  // Add the LC residuals.
  vector<ResidualDesc> lc_res_desc = AddLCResiduals(node_a, node_b);
  double chi_num =
      GetChiSquareCost(std::min(node_a, node_b), std::max(node_a, node_b));
  std::cout << "chinum: " << chi_num << std::endl;
  RemoveResiduals(lc_res_desc);
  if (chi_num < 0 || !ceres::IsFinite(chi_num)) {
    return false;
  }
  chi_squared dist(3);
  double upper_critical_value = quantile(complement(dist, certainty));
  std::cout << "Boundary: " << upper_critical_value << std::endl;
  // Upper Critical Value so must be less than to accept (i.e. are similar).
  return chi_num < upper_critical_value;
}

float Solver::GetMatchScores(SLAMNode2D& node, SLAMNode2D& keyframe) {
  // TODO figure out how this even happened
  if (node.node_idx == keyframe.node_idx) {
    return 0.0f;
  }

  laser_scan_matcher::MatchLaserScans srv;
  srv.request.scan = node.lidar_factor.scan;
  srv.request.alt_scan = keyframe.lidar_factor.scan;

  int response = matcher_client.call(srv);
  if (response) {
    float score = srv.response.match_prob;
    std::cout << "Match Score between " << node.node_idx << " and " << keyframe.node_idx << ": " << score << std::endl;
    return score;
  } else {
    std::cerr << "Failed to call service match_laser_scans: "  << response  << "\t" << srv.response.match_prob << std::endl;
    exit(100);
  }
}

std::pair<double, double> Solver::GetLocalUncertaintyEstimate(const uint64_t node_idx) {
  local_uncertainty_estimator::EstimateLocalUncertainty srv;
  srv.request.scan = problem_.nodes[node_idx].lidar_factor.scan;

  int response = local_uncertainty_client.call(srv);
  if (response) {
    return std::pair<double, double>(srv.response.condition_num, srv.response.scale);
  } else {
    std::cerr << "Failed to call service estimate_local_uncertainty: "  << response << std::endl;
    exit(100);
  }
}

double Solver::GetChiSquareCost(uint64_t node_a, uint64_t node_b) {
  std::cout << "Between " << node_a << " and " << node_b << std::endl;
  CHECK_LT(node_a, solution_.size());
  CHECK_LT(node_b, solution_.size());
  CHECK_GT(solution_.size(), 1);
  double* param_block_a = solution_[node_a].pose;
  double* param_block_b = solution_[node_b].pose;
  // Set the first LC pose constant so it has 0 covariance with itself.
  // Then the covariance obtained from all the other poses will be relative to
  // this pose. Also set the first one variable.
  // Grab the covariance between these two blocks.
  ceres::Covariance::Options cov_options;
  cov_options.num_threads = std::thread::hardware_concurrency();
  ceres::Covariance covariance(cov_options);
  vector<pair<const double*, const double*>> param_blocks;
  param_blocks.push_back(std::make_pair(param_block_a, param_block_b));
  // TODO: Remove after confirming works as intended
  param_blocks.push_back(std::make_pair(param_block_a, param_block_a));
  param_blocks.push_back(std::make_pair(param_block_b, param_block_b));
  std::cout << "Computing Covariance" << std::endl;
  CHECK(covariance.Compute(param_blocks,
                           (ceres::Problem*)ceres_information.problem.get()));
  double covariance_ab[3 * 3];
  CHECK(covariance.GetCovarianceBlock(param_block_a, param_block_b,
                                      covariance_ab));
  // Now get the expected transformation from CSM.
  // TODO: Should we be restricting by rotation here?
  std::cout << "Running CSM" << std::endl;
  std::pair<double, std::pair<Vector2f, float>> trans_prob_pair =
      scan_matcher.GetTransformation(
          problem_.nodes[node_a].lidar_factor.pointcloud,
          problem_.nodes[node_b].lidar_factor.pointcloud,
          solution_[node_a].pose[2], solution_[node_b].pose[2],
          math_util::DegToRad(90));
  auto trans = trans_prob_pair.second;
  double difference_from_solution[3];
  std::cout << "Pose A: " << param_block_a[0] << " " << param_block_a[1] << " "
            << param_block_a[2] << std::endl;
  std::cout << "Pose B: " << param_block_b[0] << " " << param_block_b[1] << " "
            << param_block_b[2] << std::endl;
  std::cout << "Difference from poses: " << param_block_a[0] - param_block_b[0]
            << " " << param_block_a[1] - param_block_b[1] << " "
            << param_block_a[2] - param_block_b[2] << std::endl;
  std::cout << "Raw CSM Translation: " << std::endl << trans.first << std::endl;
  std::cout << "Raw CSM Rotation: " << trans.second << std::endl;
  // TODO: Trying Raw CSM as it makes more sense.
  difference_from_solution[0] =
      param_block_a[0] - (param_block_b[0] + trans.first.x());
  difference_from_solution[1] =
      param_block_a[1] - (param_block_b[1] + trans.first.y());
  difference_from_solution[2] =
      param_block_a[2] - (param_block_b[2] + trans.second);
  std::cout << "CSM Found Difference from Solution: " << std::endl
            << difference_from_solution[0] << " " << difference_from_solution[1]
            << " " << difference_from_solution[2] << " " << std::endl;
  Eigen::Vector3d vec = Eigen::Map<Eigen::Vector3d>(difference_from_solution);
  Eigen::Matrix3d cov = Eigen::Map<Eigen::Matrix3d>(covariance_ab);
  std::cout << "residuals:\n" << vec << std::endl;
  std::cout << "covariance:\n" << cov << std::endl;
  double cost = (vec.transpose() * cov.inverse() * vec);
  return cost;
}

void Solver::LoadSLAMSolution(const char* poses_path) {
  std::map<double, Vector3f> poses;
  std::ifstream poses_file;
  poses_file.open(poses_path);
  if(poses_file.is_open()) {
    double timestamp;
    float pose_x, pose_y, theta;
    while (poses_file >> timestamp >> pose_x >> pose_y >> theta) {
      poses[timestamp] = Vector3f(pose_x, pose_y, theta);
    }
  }
  poses_file.close();
  std::cout << "Finished loading solution file." << std::endl;
  for(size_t i = 0; i < solution_.size(); i++) {
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

void Solver::WriteCallback(const WriteMsgConstPtr& msg) {
  if (config_.CONFIG_pose_output_file.compare("") == 0) {
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

void Solver::Vectorize(const WriteMsgConstPtr& msg) {
  std::cout << "Vectorizing" << std::endl;
  using VectorMaps::LineSegment;
  vector<Vector2f> whole_pointcloud;
  for (const SLAMNode2D& n : problem_.nodes) {
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
  for (const LineSegment& line : lines) {
    Vector3f line_start(line.start_point.x(), line.start_point.y(), 0.0);
    Vector3f line_end(line.end_point.x(), line.end_point.y(), 0.0);
    gui_helpers::AddLine(line_start, line_end, gui_helpers::Color4f::kWhite,
                         &line_mark);
  }
  
  std::cout << "Created map: Pointcloud size: " << whole_pointcloud.size() << "\tLines size: " << lines.size() << std::endl;
  
  if (config_.CONFIG_map_output_file.compare("") != 0) {
    std::cout << "Writing map to file..." << std::endl;
    std::ofstream output_file;
    output_file.open(config_.CONFIG_map_output_file);
    for(auto line : lines) {
      output_file << line.start_point.x() << "," << line.start_point.y() << "," << line.end_point.x() << "," << line.end_point.y() << std::endl;
    }
    output_file.close();
  }

  std::cout << "Publishing map..." << std::endl;
  for (int i = 0; i < 5; i++) {
    lines_pub.publish(line_mark);
    sleep(1);
  }
}


