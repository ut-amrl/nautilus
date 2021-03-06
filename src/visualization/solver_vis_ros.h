//
// Created by jack on 10/15/20.
//

#ifndef NAUTILUS_SOLVER_VIS_IMPL_H
#define NAUTILUS_SOLVER_VIS_IMPL_H

#include "../util/slam_types.h"
#include "./solver_vis.h"
#include "ros/ros.h"

namespace nautilus::visualization {

class SolverVisualizerROS : public SolverVisualizer {
 public:
  SolverVisualizerROS(std::shared_ptr<slam_types::SLAMState2D> state,
                      ros::NodeHandle& n);
  void DrawSolution() const;
  void DrawCorrespondence(const PointCorrespondences&) const;
  void DrawScans(const std::vector<size_t> scans) const;
  void DrawCovariances(std::vector<std::tuple<size_t, Eigen::Matrix2f>>) const;

 private:
  ros::Publisher points_pub_;
  ros::Publisher poses_pub_;
  ros::Publisher edge_pub_;
  ros::Publisher planar_pub_;
  ros::Publisher correspondence_pub_;
  ros::Publisher scan_pub_;
  ros::Publisher covariance_pub_;
};

}  // namespace nautilus::visualization

#endif  // NAUTILUS_SOLVER_VIS_IMPL_H
