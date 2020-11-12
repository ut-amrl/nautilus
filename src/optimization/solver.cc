// Created by jack on 9/25/19.
//

#include "./solver.h"

#include <visualization_msgs/Marker.h>

#include <algorithm>
#include <fstream>
#include <thread>
#include <vector>

#include "../util/kdtree.h"
#include "../util/math_util.h"
#include "../util/timer.h"
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
using slam_types::SLAMState2D;
using std::pair;
using std::vector;
using visualization::SolverVisualizer;

Solver::Solver(ros::NodeHandle &n, std::shared_ptr<SLAMState2D> state,
               std::unique_ptr<SolverVisualizer> vis)
    : n_(n), scan_matcher(30, 2, 0.3, 0.01), state_(state) {
  vis_.swap(vis);
  matcher_client = n_.serviceClient<MatchLaserScans>("match_laser_scans");
  local_uncertainty_client =
      n_.serviceClient<EstimateLocalUncertainty>("estimate_local_uncertainty");
}

/*----------------------------------------------------------------------------*
 *                        POINT MATCHING FUNCTIONS                            |
 *----------------------------------------------------------------------------*/

Vector2f GetPointNormal(const Vector2f &point,
                        std::shared_ptr<KDTree<float, 2>> lookup_tree) {
  KDNodeValue<float, 2> point_in_tree;
  double distance_to_point =
      lookup_tree->FindNearestPoint(point, 0.00001, &point_in_tree);
  // Points should be within a very small distance to themselves, as this is the
  // same point. If this fails, it means the point is not in this tree and you
  // should not lookup the normal using this function.
  CHECK_LT(distance_to_point, 0.00001);
  return point_in_tree.normal;
}

std::optional<Vector2f> FindClosestPoint(
    const Vector2f &source_point,
    std::shared_ptr<KDTree<float, 2>> match_lookup_tree) {
  KDNodeValue<float, 2> closest_match;
  double dist = match_lookup_tree->FindNearestPoint(
      source_point, SolverConfig::CONFIG_outlier_threshold, &closest_match);
  if (dist < SolverConfig::CONFIG_outlier_threshold) {
    return std::optional(closest_match.point);
  }
  return std::nullopt;
}

std::tuple<vector<Vector2f>, std::shared_ptr<KDTree<float, 2>>>
GetPointsAndLookupTree(const std::shared_ptr<slam_types::SLAMState2D> state,
                       int source_node_idx, int target_node_idx,
                       const PointcloudType &type) {
  vector<Vector2f> source_points;
  std::shared_ptr<KDTree<float, 2>> match_lookup_tree;
  switch (type) {
    case PointcloudType::PLANAR:
      match_lookup_tree =
          state->problem.nodes[target_node_idx].lidar_factor.planar_tree;
      source_points =
          state->problem.nodes[source_node_idx].lidar_factor.planar_points;
      break;
    case PointcloudType::EDGE:
      match_lookup_tree =
          state->problem.nodes[target_node_idx].lidar_factor.edge_tree;
      source_points =
          state->problem.nodes[source_node_idx].lidar_factor.edge_points;
      break;
    case PointcloudType::ALL:
      match_lookup_tree =
          state->problem.nodes[target_node_idx].lidar_factor.pointcloud_tree;
      source_points =
          state->problem.nodes[source_node_idx].lidar_factor.pointcloud;
      break;
    default:
      throw "Unknown PointcloudType";
      break;
  }
  return {source_points, match_lookup_tree};
}

/// @desc: Returns a matching between source and target nodes specified
/// pointclouds, the matching is found by finding the closest target point to
/// every source point within the CONFIG::outlier_threshold.
/// @param source_node_idx: The index of the first node in the state, points
/// will be matched against target.
/// @param target_node_idx: The index of the second node in the state, points
/// will be used for finding closest match to source point.
/// @returns A PointCorrespondences object which holds the matches between
/// source and target, maybe empty.
PointCorrespondences Solver::GetPointToPointMatching(
    int source_node_idx, int target_node_idx, const PointcloudType &type) {
  CHECK_LT(source_node_idx, state_->problem.nodes.size());
  CHECK_LT(target_node_idx, state_->problem.nodes.size());
  CHECK_LT(source_node_idx, state_->solution.size());
  CHECK_LT(target_node_idx, state_->solution.size());
  // Get the right pointclouds for match lookup based on the PointcloudType
  // given.
  auto [source_points, match_lookup_tree] =
      GetPointsAndLookupTree(state_, source_node_idx, target_node_idx, type);
  CHECK_NOTNULL(match_lookup_tree);
  auto source_pose = state_->solution[source_node_idx].pose;
  auto target_pose = state_->solution[target_node_idx].pose;
  PointCorrespondences correspondence(source_pose, target_pose, source_node_idx,
                                      target_node_idx);
  // Affine transformations from the two pose's reference frames.
  Affine2f source_to_world = GetPoseAsAffine<float>(state_, source_node_idx);
  Affine2f target_to_world = GetPoseAsAffine<float>(state_, target_node_idx);
  // The full pointcloud trees are always used for finding normals, as they are
  // the most accurate.
  auto source_normal_tree =
      state_->problem.nodes[source_node_idx].lidar_factor.pointcloud_tree;
  auto target_normal_tree =
      state_->problem.nodes[target_node_idx].lidar_factor.pointcloud_tree;
  for (const Vector2f &source_point : source_points) {
    // Transform the source point to the target frame.
    Vector2f source_point_target_frame =
        target_to_world.inverse() * source_to_world * source_point;
    if (auto match =
            FindClosestPoint(source_point_target_frame, match_lookup_tree);
        match.has_value()) {
      // We have a match! So we get the normals and save it.
      Vector2f source_normal = GetPointNormal(source_point, source_normal_tree);
      Vector2f target_normal =
          GetPointNormal(match.value(), target_normal_tree);
      correspondence.AddCorrespondence(source_point, source_normal,
                                       match.value(), target_normal);
    }
  }
  return correspondence;
}

/// @desc: Finds the point in the match_lookup_tree that is closest to the
/// source point and has a "similar" normal. A "similar" normal is when the two
/// normals have angles between them of less than 20 degrees.
std::optional<Vector2f> FindClosestPointWithSimilarNormal(
    const Vector2f &source_point, const Vector2f &source_normal,
    std::shared_ptr<KDTree<float, 2>> match_lookup_tree,
    const double outlier_threshold) {
  std::vector<KDNodeValue<float, 2>> possible_matches;
  match_lookup_tree->FindNeighborPoints(
      source_point, SolverConfig::CONFIG_outlier_threshold, &possible_matches);
  // Sort possible_matches by the distance from the source_point, as this is not
  // guaranteed by the KDTree.
  std::sort(possible_matches.begin(), possible_matches.end(),
            [&source_point](KDNodeValue<float, 2> point_1,
                            KDNodeValue<float, 2> point_2) {
              return (source_point - point_1.point).norm() <
                     (source_point - point_2.point).norm();
            });
  // Find the match point with the closest normal.
  for (auto current_match : possible_matches) {
    if (NormalsSimilar(current_match.normal, source_normal,
                       cos(math_util::DegToRad(20.0)))) {
      return std::optional(current_match.point);
    }
  }
  return std::nullopt;
}

/// @desc: Similar to the FindClosestPointWithSimilarNormal above, except
/// instead of using a static outlier threshold. it grows the threshold up the
/// max threshold so that the runtime is faster along the way.
std::optional<Vector2f> FindClosestPointWithSimilarNormal(
    const Vector2f &source_point, const Vector2f &source_normal,
    std::shared_ptr<KDTree<float, 2>> match_lookup_tree) {
  for (int i = 6; i > 0; i--) {
    double outlier_threshold =
        SolverConfig::CONFIG_outlier_threshold / static_cast<double>(i);
    auto match = FindClosestPointWithSimilarNormal(
        source_point, source_normal, match_lookup_tree, outlier_threshold);
    if (match.has_value()) {
      return match;
    }
  }
  return std::nullopt;
}

PointCorrespondences Solver::GetPointToNormalMatching(
    int source_node_idx, int target_node_idx, const PointcloudType &type) {
  CHECK_LT(source_node_idx, state_->problem.nodes.size());
  CHECK_LT(target_node_idx, state_->problem.nodes.size());
  CHECK_LT(source_node_idx, state_->solution.size());
  CHECK_LT(target_node_idx, state_->solution.size());
  // Get the right pointclouds for match lookup based on the PointcloudType
  // given.
  auto [source_points, match_lookup_tree] =
      GetPointsAndLookupTree(state_, source_node_idx, target_node_idx, type);
  CHECK_NOTNULL(match_lookup_tree);
  auto source_pose = state_->solution[source_node_idx].pose;
  auto target_pose = state_->solution[target_node_idx].pose;
  PointCorrespondences correspondence(source_pose, target_pose, source_node_idx,
                                      target_node_idx);
  // Affine transformations from the two pose's reference frames.
  Affine2f source_to_world = GetPoseAsAffine<float>(state_, source_node_idx);
  Affine2f target_to_world = GetPoseAsAffine<float>(state_, target_node_idx);
  // The full pointcloud trees are always used for finding normals, as they are
  // the most accurate.
  auto source_normal_tree =
      state_->problem.nodes[source_node_idx].lidar_factor.pointcloud_tree;
  auto target_normal_tree =
      state_->problem.nodes[target_node_idx].lidar_factor.pointcloud_tree;
  for (const Vector2f &source_point : source_points) {
    // Transform the source point to the target frame.
    Vector2f source_point_target_frame =
        target_to_world.inverse() * source_to_world * source_point;
    Vector2f source_normal = GetPointNormal(source_point, source_normal_tree);
    if (auto match = FindClosestPointWithSimilarNormal(
            source_point_target_frame, source_normal, match_lookup_tree);
        match.has_value()) {
      // We have a match! So we get the normals and save it.
      Vector2f target_normal =
          GetPointNormal(match.value(), target_normal_tree);
      correspondence.AddCorrespondence(source_point, source_normal,
                                       match.value(), target_normal);
    }
  }
  return correspondence;
}

/*----------------------------------------------------------------------------*
 *                          SLAM SOLVING FUNCTIONS                            |
 *----------------------------------------------------------------------------*/

void AddNormalCorrespondence(CeresInformation *ceres_info,
                             const PointCorrespondences &corr) {
  if (!corr.source_points.empty()) {
    ceres_info->problem->AddResidualBlock(
        LIDARNormalResidual::create(corr.source_points, corr.target_points,
                                    corr.source_normals, corr.target_normals),
        NULL, corr.source_pose, corr.target_pose);
  }
}

void AddPointCorrespondence(CeresInformation *ceres_info,
                            const PointCorrespondences &corr) {
  if (!corr.source_points.empty()) {
    ceres_info->problem->AddResidualBlock(
        LIDARPointResidual::create(corr.source_points, corr.target_points,
                                   corr.source_normals, corr.target_normals),
        NULL, corr.source_pose, corr.target_pose);
  }
}

double Solver::BuildOptimizationOverWindow(
    int64_t window_size, const OptimizationType &optimization_type) {
  double difference = 0.0;
  for (size_t node_i_index = 0; node_i_index < state_->problem.nodes.size();
       node_i_index++) {
    for (size_t node_j_index =
             std::max((int64_t)(node_i_index)-window_size, 0l);
         node_j_index < node_i_index; node_j_index++) {
      if (optimization_type == OptimizationType::FEATURE) {
        // Planar Correspondences
        auto planar_correspondence = GetPointToPointMatching(
            node_i_index, node_j_index, PointcloudType::PLANAR);
        difference += planar_correspondence.difference;
        // Only add if we got matches between pointclouds.
        AddNormalCorrespondence(&ceres_information, planar_correspondence);
        vis_->DrawCorrespondence(planar_correspondence);
        // Edge Correspondences
        auto edge_correspondence = GetPointToPointMatching(
            node_i_index, node_j_index, PointcloudType::EDGE);
        difference += edge_correspondence.difference;
        AddPointCorrespondence(&ceres_information, edge_correspondence);
        vis_->DrawCorrespondence(edge_correspondence);
      } else {
        // Correspondence involving all the points. Using Normals for more
        // accurate results.
        auto correspondence = GetPointToPointMatching(
            node_i_index, node_j_index, PointcloudType::ALL);
        difference += correspondence.difference;
        AddPointCorrespondence(&ceres_information, correspondence);
        vis_->DrawCorrespondence(correspondence);
      }
    }
  }
  return difference;
}

void Solver::OptimizeOverGrowingWindow(const OptimizationType &type,
                                       const ceres::Solver::Options &options) {
  // While our solution moves more than the stopping_accuracy,
  // continue to optimize.
  for (int64_t window_size = SolverConfig::CONFIG_lidar_constraint_amount_min;
       window_size <= SolverConfig::CONFIG_lidar_constraint_amount_max;
       window_size++) {
    LOG(INFO) << "Using window size: " << window_size << std::endl;
    ceres_information.ResetProblem();
    // Add all the odometry constraints between our poses.
    AddOdomFactors(ceres_information.problem.get(),
                   state_->problem.odometry_factors,
                   SolverConfig::CONFIG_translation_weight,
                   SolverConfig::CONFIG_rotation_weight);
    // Add the vision factors.
    BuildOptimizationOverWindow(window_size, type);
    // Normalize the difference so it's an average over each node.
    AddHITLResiduals(ceres_information.problem.get());
    ceres::Solver::Summary summary;
    ceres::Solve(options, ceres_information.problem.get(), &summary);
  }
}

void Solver::SolveSLAM() {
  // Setup ceres for evaluation of the problem.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  options.callbacks.push_back(
      dynamic_cast<ceres::IterationCallback *>(vis_.get()));
  // Draw the initial solution
  vis_->DrawSolution();
  // Optimize the first time with just features, for faster performance.
  OptimizeOverGrowingWindow(OptimizationType::FEATURE, options);
  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i < 5; i++) {
    vis_->DrawSolution();
  }
}

void Solver::AddOdomFactors(ceres::Problem *ceres_problem,
                            vector<OdometryFactor2D> factors,
                            double trans_weight, double rot_weight) {
  for (const OdometryFactor2D &odom_factor : factors) {
    CHECK_LT(odom_factor.pose_i, odom_factor.pose_j);
    CHECK_GT(state_->solution.size(), odom_factor.pose_i);
    CHECK_GT(state_->solution.size(), odom_factor.pose_j);
    ceres::ResidualBlockId id = ceres_problem->AddResidualBlock(
        OdometryResidual::create(odom_factor, trans_weight, rot_weight), NULL,
        state_->solution[odom_factor.pose_i].pose,
        state_->solution[odom_factor.pose_j].pose);
    ceres_information.res_descriptors.emplace_back(odom_factor.pose_i,
                                                   odom_factor.pose_j, id);
  }
  if (state_->solution.size() > 0) {
    ceres_problem->SetParameterBlockConstant(state_->solution[0].pose);
  }
}

OdometryFactor2D Solver::GetDifferenceOdom(const uint64_t node_a,
                                           const uint64_t node_b) {
  double *pose_a = state_->solution[node_a].pose;
  double *pose_b = state_->solution[node_b].pose;
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
  CHECK_GT(state_->solution.size(), 1);
  vector<OdometryFactor2D> factors;
  for (uint64_t index = 1; index < state_->solution.size(); index++) {
    // Get the change in translation.
    for (uint64_t prev_idx =
             std::max((uint64_t)0,
                      index - SolverConfig::CONFIG_lidar_constraint_amount_max);
         prev_idx < index; prev_idx++) {
      Vector2f prev_loc(state_->solution[prev_idx].pose[0],
                        state_->solution[prev_idx].pose[1]);
      Vector2f loc(state_->solution[index].pose[0],
                   state_->solution[index].pose[1]);

      double rot_change =
          state_->solution[index].pose[2] - state_->solution[prev_idx].pose[2];
      Vector2f trans_change = loc - prev_loc;
      factors.emplace_back(prev_idx, index, trans_change, rot_change);
    }
  }
  return factors;
}

vector<OdometryFactor2D> Solver::GetSolvedOdomFactorsBetweenNodes(uint64_t a,
                                                                  uint64_t b) {
  CHECK_GT(state_->solution.size(), b);
  CHECK_GT(b, a);
  vector<OdometryFactor2D> factors;
  for (uint64_t index = a + 1; index <= b; index++) {
    // Get the change in translation.
    uint64_t prev_idx = index - 1;
    Vector2f prev_loc(state_->solution[prev_idx].pose[0],
                      state_->solution[prev_idx].pose[1]);
    Vector2f loc(state_->solution[index].pose[0],
                 state_->solution[index].pose[1]);

    double rot_change = math_util::AngleDiff(
        state_->solution[index].pose[2], state_->solution[prev_idx].pose[2]);
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
  for (size_t node_idx = 0; node_idx < state_->problem.nodes.size();
       node_idx++) {
    vector<Vector2f> points_on_a;
    vector<Vector2f> points_on_b;
    double *pose_ptr = state_->solution[node_idx].pose;
    Affine2f node_to_world =
        PoseArrayToAffine(&pose_ptr[2], &pose_ptr[0]).cast<float>();
    for (const Vector2f &point :
         state_->problem.nodes[node_idx].lidar_factor.pointcloud) {
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
      CHECK_LT(a_pose.node_idx, state_->solution.size());
      problem->AddResidualBlock(
          PointToLineResidual::create(constraint.line_a, pointcloud_a), NULL,
          state_->solution[a_pose.node_idx].pose, constraint.chosen_line_pose);
    }
    for (const LCPose &b_pose : constraint.line_b_poses) {
      const vector<Vector2f> &pointcloud_b = b_pose.points_on_feature;
      CHECK_LT(b_pose.node_idx, state_->solution.size());
      problem->AddResidualBlock(
          PointToLineResidual::create(constraint.line_a, pointcloud_b), NULL,
          state_->solution[b_pose.node_idx].pose, constraint.chosen_line_pose);
    }
  }
}

void Solver::HitlCallback(const HitlSlamInputMsgConstPtr &hitl_ptr) {
  state_->problem.odometry_factors = GetSolvedOdomFactors();
  const HitlSlamInputMsg hitl_msg = *hitl_ptr;
  // Get the poses that belong to this input.
  const HitlLCConstraint colinear_constraint =
      GetRelevantPosesForHITL(hitl_msg);
  std::cout << "Found " << colinear_constraint.line_a_poses.size()
            << " poses for the first line." << std::endl;
  std::cout << "Found " << colinear_constraint.line_b_poses.size()
            << " poses for the second line." << std::endl;
  // TODO: Re-add constraint visualization
  hitl_constraints_.push_back(colinear_constraint);
  // Resolve the initial problem with extra pointcloud residuals between these
  // loop closed points.
  // TODO: Find a better way to set these up.
  //  translation_weight_ = lc_translation_weight_;
  //  rotation_weight_ = lc_rotation_weight_;
  std::cout << "Solving problem with HITL constraints..." << std::endl;
  SolveSLAM();
  // TODO: This is giving worse results.
  state_->problem.odometry_factors = initial_odometry_factors;
  std::cout << "Solving problem with initial odometry constraints..."
            << std::endl;
  SolveSLAM();
  std::cout << "Waiting for Loop Closure input." << std::endl;
}

/*----------------------------------------------------------------------------*
 *                            MISC. SOLVER CALLBACKS                          |
 *----------------------------------------------------------------------------*/

void Solver::WriteCallback(const WriteMsgConstPtr &msg) {
  if (SolverConfig::CONFIG_pose_output_file.compare("") == 0) {
    std::cout << "No output file specified, not writing!" << std::endl;
    return;
  }
  std::cout << "Writing Poses" << std::endl;
  std::ofstream output_file;
  output_file.open(SolverConfig::CONFIG_pose_output_file);
  for (const SLAMNodeSolution2D &sol_node : state_->solution) {
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
  for (const SLAMNode2D &n : state_->problem.nodes) {
    vector<Vector2f> pc = n.lidar_factor.pointcloud;
    pc = TransformPointcloud(state_->solution[n.node_idx].pose, pc);
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
  }
}

/*----------------------------------------------------------------*
 |                      Auto LC Functions                         |
 *----------------------------------------------------------------*/

Eigen::Vector2f ComputeMean(const vector<Vector2f> &pointcloud) {
  // Compute the mean and return the mean vector.
  Eigen::Vector2f mean_vector(0, 0);
  for (const auto &p : pointcloud) {
    mean_vector += p;
  }
  return (1.0 / pointcloud.size()) * mean_vector;
}

/// @desc: Computes the min eigenvalue / max eigenvalue of a scatter matrix for
/// a particular scan.
double ComputeScatterMatrixScore(const vector<Vector2f> &pointcloud) {
  Vector2f mean = ComputeMean(pointcloud);
  Eigen::Matrix2f scatter_matrix;
  scatter_matrix << 0, 0, 0, 0;
  // Compute the scatter matrix.
  for (const auto &p : pointcloud) {
    scatter_matrix += (p - mean) * (p - mean).transpose();
  }
  Eigen::EigenSolver<Eigen::Matrix2f> eigen_solver;
  eigen_solver.compute(scatter_matrix);
  // Now extract the eigen values.
  auto eigen_values = eigen_solver.eigenvalues();
  CHECK_EQ(eigen_values.rows(), 2);
  double ev_1 = eigen_values(0, 0).real();
  double ev_2 = eigen_values(1, 0).real();
  return std::min(ev_1, ev_2) / std::max(ev_1, ev_2);
}

bool DistantFromLastScan(std::shared_ptr<slam_types::SLAMState2D> state,
                         int node_idx, std::vector<int> scans,
                         double distance) {
  if (scans.empty()) {
    return true;
  }
  auto last_scan_trans = GetPoseTranslation(state, scans[scans.size() - 1]);
  auto node_trans = GetPoseTranslation(state, node_idx);
  return (node_trans - last_scan_trans).norm() >= distance;
}

vector<int> Solver::GetScansForLC() {
  vector<int> scans;
  for (size_t i = 0; i < state_->problem.nodes.size(); i++) {
    if (!DistantFromLastScan(state_, static_cast<int>(i), scans, 5)) {
      // Skip close scans.
      continue;
    }
    double score = ComputeScatterMatrixScore(
        state_->problem.nodes[i].lidar_factor.pointcloud);
    // A score close to 1 means a good spread in both axes of the scan.
    // So a high score means a good location for loop closure because it has
    // good spread.
    if (score >= 0.70) {
      scans.push_back(i);
    }
  }
  return scans;
}

// Gets the Covariance matrix between any two scans.
Eigen::Matrix2f GetCovarianceMatrix(std::shared_ptr<slam_types::SLAMState2D> state,
                                    ceres::Problem *problem, int source, int target) {
  ceres::Covariance::Options options;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  ceres::Covariance cov(options);
  vector<std::pair<const double *, const double *>> blocks;
  blocks.emplace_back(state->solution[source].pose,
                      state->solution[target].pose);
  double values[9] = {0};
  problem->SetParameterBlockVariable(state->solution[0].pose);
  problem->SetParameterBlockConstant(state->solution[source].pose);
  CHECK(cov.Compute(blocks, problem));
  cov.GetCovarianceBlock(state->solution[source].pose,
                         state->solution[target].pose, values);
  problem->SetParameterBlockVariable(state->solution[source].pose);
  problem->SetParameterBlockConstant(state->solution[0].pose);
  Eigen::Matrix2d covariance_mat; //= Eigen::Map<Eigen::Matrix3d>(values);
  covariance_mat << values[0], values[1], values[3], values[4];
  return covariance_mat.cast<float>();
}

// Gets the score between the source and the target node. A representation of
// how likely they are to match for loop closure.
double ChiSquareScore(std::shared_ptr<slam_types::SLAMState2D> state,
                      ceres::Problem *problem, int source, int target) {
  auto covariance_mat = GetCovarianceMatrix(state, problem, source, target);
  Eigen::Vector2f source_pose = GetPoseTranslation(state, source);
  Eigen::Vector2f target_pose = GetPoseTranslation(state, target);
  return (target_pose - source_pose).transpose() * covariance_mat.inverse() *
         (target_pose - source_pose);
}


// Finds the scan most likely to be a match, if there exists one.
std::tuple<double, int> Solver::BestScanMatch(int source_scan,
                                              std::vector<int> scans) {
  CHECK_GT(scans.size(), 0);
  double best_score = std::numeric_limits<double>::max();
  int best_match = scans[0];
  for (int target_scan : scans) {
    if (source_scan == target_scan) {
      continue;
    }
    auto score = ChiSquareScore(state_, ceres_information.problem.get(),
                                source_scan, target_scan);
    if (score < best_score) {
      best_match = target_scan;
      best_score = score;
    }
  }
  return {best_score, best_match};
}

void Solver::SolveAutoLC() {
  auto scans = GetScansForLC();
  vis_->DrawScans(scans);
  // TODO: Remove later, using correspondence to draw matches.
  double empty_pose[3] = {0};
  PointCorrespondences corr(empty_pose, empty_pose, 0, 0);
  // Now loop through and find if it matches any of the scans.
  for (int i : scans) {
    auto [score, match] = BestScanMatch(i, scans);
    std::cout << "Matched Scan #" << i << " with Scan #" << match
              << " with score " << score << std::endl;
    corr.source_points.push_back(GetPoseTranslation(state_, i));
    corr.target_points.push_back(GetPoseTranslation(state_, match));
    // TODO: Remove later, using to draw the covariances.
    vector<std::tuple<int, Eigen::Matrix2f>> covs;
    covs.emplace_back(i, GetCovarianceMatrix(state_, ceres_information.problem.get(), i, match));
    covs.emplace_back(match, GetCovarianceMatrix(state_, ceres_information.problem.get(), match, i));
    vis_->DrawCovariances(covs);
  }
  vis_->DrawCorrespondence(corr);
}

}  // namespace nautilus
