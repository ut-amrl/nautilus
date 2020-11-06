//
// Created by jack on 9/25/19.
//

#ifndef SRC_SOLVER_H_
#define SRC_SOLVER_H_

#include <boost/math/distributions/chi_squared.hpp>
#include <vector>

#include "../input/pointcloud_helpers.h"
#include "../util/gui_helpers.h"
#include "../util/kdtree.h"
#include "../util/slam_types.h"
#include "../util/slam_util.h"
#include "../visualization/solver_vis.h"
#include "./data_structures.h"
#include "CorrelativeScanMatcher.h"
#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "config_reader/config_reader.h"
#include "geometry_msgs/PoseArray.h"
#include "glog/logging.h"
#include "nautilus/HitlSlamInputMsg.h"
#include "nautilus/WriteMsg.h"
#include "ros/node_handle.h"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"

namespace nautilus {
namespace SolverConfig {
CONFIG_DOUBLE(translation_weight, "translation_weight");
CONFIG_DOUBLE(rotation_weight, "rotation_weight");
CONFIG_DOUBLE(lc_translation_weight, "lc_translation_weight");
CONFIG_DOUBLE(lc_rotation_weight, "lc_rotation_weight");
CONFIG_DOUBLE(lc_base_max_range, "lc_base_max_range");
CONFIG_DOUBLE(lc_max_range_scaling, "lc_max_range_scaling");
CONFIG_STRING(lc_debug_output_dir, "lc_debug_output_dir");
CONFIG_STRING(pose_output_file, "pose_output_file");
CONFIG_STRING(map_output_file, "map_output_file");
CONFIG_DOUBLE(accuracy_change_stop_threshold, "accuracy_change_stop_threshold");
CONFIG_DOUBLE(max_lidar_range, "max_lidar_range");
CONFIG_DOUBLE(lc_match_threshold, "lc_match_threshold");
CONFIG_INT(lidar_constraint_amount_min, "lidar_constraint_amount_min");
CONFIG_INT(lidar_constraint_amount_max, "lidar_constraint_amount_max");
CONFIG_DOUBLE(outlier_threshold, "outlier_threshold");
CONFIG_DOUBLE(hitl_line_width, "hitl_line_width");
CONFIG_INT(hitl_pose_point_threshold, "hitl_pose_point_threshold");
// Auto LC configs
CONFIG_BOOL(keyframe_local_uncertainty_filtering,
            "keyframe_local_uncertainty_filtering");
CONFIG_BOOL(keyframe_chi_squared_test, "keyframe_chi_squared_test");
CONFIG_DOUBLE(keyframe_min_odom_distance, "keyframe_min_odom_distance");
CONFIG_DOUBLE(local_uncertainty_condition_threshold,
              "local_uncertainty_condition_threshold");
CONFIG_DOUBLE(local_uncertainty_scale_threshold,
              "local_uncertainty_scale_threshold");
CONFIG_INT(local_uncertainty_prev_scans, "local_uncertainty_prev_scans");
CONFIG_INT(lc_match_window_size, "lc_match_window_size");
CONFIG_INT(lc_min_keyframes, "lc_min_keyframes");
CONFIG_DOUBLE(csm_score_threshold, "csm_score_threshold");
CONFIG_DOUBLE(translation_std_dev, "translation_standard_deviation");
CONFIG_DOUBLE(rotation_std_dev, "rotation_standard_deviation");
};  // namespace SolverConfig

/*----------------------------------------------------------------------------*
 *                                SOLVER                                      |
 *----------------------------------------------------------------------------*/

enum class PointcloudType {
    PLANAR,
    EDGE,
    ALL
};

class Solver {
 public:
  Solver(ros::NodeHandle& n, std::shared_ptr<slam_types::SLAMState2D> state,
         std::unique_ptr<visualization::SolverVisualizer> vis);
  void SolveSLAM();
  std::vector<slam_types::SLAMNodeSolution2D> SolvePoseSLAM();
      double GetPointCorrespondences(
      const vector<Eigen::Vector2f> source_pointcloud,
      const std::shared_ptr<KDTree<float, 2>> source_tree,
      const vector<Eigen::Vector2f> target_pointcloud,
      const std::shared_ptr<KDTree<float, 2>> target_tree,
      const std::shared_ptr<KDTree<float, 2>> target_norm_tree,
      double* source_pose, double* target_pose,
      PointCorrespondences* point_correspondences);
  double GetPointCorrespondencesByNormal(
      const vector<Eigen::Vector2f> source_pointcloud,
      const std::shared_ptr<KDTree<float, 2>> source_tree,
      const vector<Eigen::Vector2f> target_pointcloud,
      const std::shared_ptr<KDTree<float, 2>> target_tree,
      const std::shared_ptr<KDTree<float, 2>> norm_tree, double* source_pose,
      double* target_pose, PointCorrespondences* point_correspondences);
  void AddOdomFactors(ceres::Problem* ceres_problem,
                      std::vector<slam_types::OdometryFactor2D> factors,
                      double trans_weight, double rot_weight);
  void HitlCallback(const nautilus::HitlSlamInputMsgConstPtr& hitl_ptr);
  void WriteCallback(const nautilus::WriteMsgConstPtr& msg);
  void Vectorize(const nautilus::WriteMsgConstPtr& msg);
  std::vector<slam_types::SLAMNodeSolution2D> GetSolution() {
    return state_->solution;
  }
  HitlLCConstraint GetRelevantPosesForHITL(
      const nautilus::HitlSlamInputMsg& hitl_msg);
  void SolveForLC();
  double AddResidualsForAutoLC(ceres::Problem* problem, bool include_lidar);
  void AddPointCloudResiduals(ceres::Problem* problem);
  std::vector<slam_types::OdometryFactor2D> GetSolvedOdomFactors();

 private:
  PointCorrespondences GetPointToPointMatching(int source_node_idx, int target_node_idx, PointcloudType type);
      double CostFromResidualDescriptor(const ResidualDesc& res_desc);
  double GetChiSquareCost(uint64_t node_a, uint64_t node_b);
  slam_types::OdometryFactor2D GetDifferenceOdom(const uint64_t node_a,
                                                 const uint64_t node_b);
  slam_types::OdometryFactor2D GetDifferenceOdom(const uint64_t node_a,
                                                 const uint64_t node_b,
                                                 Eigen::Vector3f trans);
  vector<ResidualDesc> AddLCResiduals(const uint64_t node_a,
                                      const uint64_t node_b);
  void AddHITLResiduals(ceres::Problem* problem);
  void RemoveResiduals(vector<ResidualDesc> descs);
  void AddKeyframe(slam_types::SLAMNode2D& node);
  float GetMatchScores(slam_types::SLAMNode2D& node,
                       slam_types::SLAMNode2D& keyframe);
  AutoLCConstraint computeAutoLCConstraint(const uint64_t node_a,
                                           const uint64_t node_b);

  slam_types::OdometryFactor2D GetTotalOdomChange(
      const std::vector<slam_types::OdometryFactor2D>& factors);
  std::vector<slam_types::OdometryFactor2D> GetSolvedOdomFactorsBetweenNodes(
      uint64_t node_a, uint64_t node_b);
  std::pair<double, double> GetLocalUncertainty(const uint64_t node_idx);
  std::pair<double, double> GetLocalUncertaintyEstimate(
      const uint64_t node_idx);
  vector<size_t> GetMatchingKeyframeIndices(size_t keyframe_index);
  std::vector<slam_types::OdometryFactor2D> initial_odometry_factors;
  ros::NodeHandle n_;
  vector<AutoLCConstraint> auto_lc_constraints_;
  vector<HitlLCConstraint> hitl_constraints_;
  ros::ServiceClient matcher_client;
  ros::ServiceClient local_uncertainty_client;
  CorrelativeScanMatcher scan_matcher;
  CeresInformation ceres_information;
  std::shared_ptr<slam_types::SLAMState2D> state_;
  std::unique_ptr<visualization::SolverVisualizer> vis_;
};

}  // namespace nautilus

#endif  // SRC_SOLVER_H_
