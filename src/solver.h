//
// Created by jack on 9/25/19.
//

#ifndef SRC_SOLVER_H_
#define SRC_SOLVER_H_

#include <ros/node_handle.h>

#include <boost/math/distributions/chi_squared.hpp>
#include <vector>

#include "./gui_helpers.h"
#include "./kdtree.h"
#include "./pointcloud_helpers.h"
#include "./slam_types.h"
#include "./solver_datastructures.h"
#include "CorrelativeScanMatcher.h"
#include "ceres/ceres.h"
#include "config_reader/config_reader.h"
#include "eigen3/Eigen/Dense"
#include "geometry_msgs/PoseArray.h"
#include "glog/logging.h"
#include "nautilus/HitlSlamInputMsg.h"
#include "nautilus/WriteMsg.h"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"
#include "solver_residuals.h"
#include "solver_visualizer.h"
#include "visualization_msgs/Marker.h"

namespace nautilus {

struct CeresInformation {
  CeresInformation()
      : cost_valid(false),
        cost(0),
        problem(new ceres::Problem()),
        res_descriptors() {}

  void ResetProblem() { *this = CeresInformation(); }

  bool cost_valid;
  double cost;
  std::shared_ptr<ceres::Problem> problem;
  std::vector<ds::ResidualDesc> res_descriptors;
};

struct SolverConfig {
  CONFIG_DOUBLE(translation_weight, "translation_weight");
  CONFIG_DOUBLE(rotation_weight, "rotation_weight");
  CONFIG_DOUBLE(lc_translation_weight, "lc_translation_weight");
  CONFIG_DOUBLE(lc_rotation_weight, "lc_rotation_weight");
  CONFIG_DOUBLE(lc_base_max_range, "lc_base_max_range");
  CONFIG_DOUBLE(lc_max_range_scaling, "lc_max_range_scaling");
  CONFIG_STRING(lc_debug_output_dir, "lc_debug_output_dir");
  CONFIG_STRING(pose_output_file, "pose_output_file");
  CONFIG_STRING(map_output_file, "map_output_file");
  CONFIG_DOUBLE(accuracy_change_stop_threshold,
                "accuracy_change_stop_threshold");
  CONFIG_DOUBLE(max_lidar_range, "max_lidar_range");
  CONFIG_DOUBLE(lc_match_threshold, "lc_match_threshold");
  CONFIG_INT(lidar_constraint_amount_min, "lidar_constraint_amount_min");
  CONFIG_INT(lidar_constraint_amount_max, "lidar_constraint_amount_max");
  CONFIG_DOUBLE(outlier_threshold, "outlier_threshold");
  CONFIG_DOUBLE(hitl_line_width, "hitl_line_width");
  CONFIG_INT(hitl_pose_point_threshold, "hitl_pose_point_threshold");

  SolverConfig() {
    std::cout << "Solver Waiting..." << std::endl;
    config_reader::WaitForInit();
    std::cout << "--- Done Waiting ---" << std::endl;
  }
};

class Solver {
 public:
  Solver() = delete;
  Solver(ros::NodeHandle& n);
  std::vector<slam_types::SLAMNodeSolution2D> SolveSLAM();
  double GetPointCorrespondences(
      const slam_types::LidarFactor& source_lidar,
      const slam_types::LidarFactor& target_lidar,
      double* source_pose,
      double* target_pose,
      ds::PointCorrespondences* point_correspondences);
  void AddOdomFactors(ceres::Problem* ceres_problem,
                      std::vector<slam_types::OdometryFactor2D> factors,
                      double trans_weight,
                      double rot_weight);
  void HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr);
  void WriteCallback(const WriteMsgConstPtr& msg);
  void Vectorize(const WriteMsgConstPtr& msg);
  ds::HitlLCConstraint GetRelevantPosesForHITL(
      const HitlSlamInputMsg& hitl_msg);
  std::vector<slam_types::OdometryFactor2D> GetSolvedOdomFactors();
  void AddSLAMNodeOdom(slam_types::SLAMNode2D& node,
                       slam_types::OdometryFactor2D& odom_factor_to_node);
  void AddSlamNode(slam_types::SLAMNode2D& node);
  void LoadSLAMSolution(const std::string& poses_path);

 private:
  double GetChiSquareCost(uint64_t node_a, uint64_t node_b);
  void AddHITLResiduals(ceres::Problem* problem);
  slam_types::OdometryFactor2D GetTotalOdomChange(
      const std::vector<slam_types::OdometryFactor2D>& factors);
  bool SimilarScans(const uint64_t node_a,
                    const uint64_t node_b,
                    const double certainty);

  slam_types::SLAMProblem2D problem_;
  std::vector<slam_types::OdometryFactor2D> initial_odometry_factors_;
  std::vector<slam_types::SLAMNodeSolution2D> solution_;
  ros::NodeHandle n_;
  std::vector<ds::HitlLCConstraint> hitl_constraints_;
  std::unique_ptr<VisualizationCallback> vis_callback_;
  std::vector<ds::LearnedKeyframe> keyframes_;
  CorrelativeScanMatcher scan_matcher_;
  CeresInformation ceres_information_;
  SolverConfig config_;
};
}  // namespace nautilus
#endif  // SRC_SOLVER_H_
