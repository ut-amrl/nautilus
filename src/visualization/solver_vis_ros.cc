//
// Created by jack on 10/15/20.
//

#include "ros/ros.h"
#include "Eigen/Dense"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseArray.h"
#include "sensor_msgs/PointCloud2.h"

#include "../util/slam_types.h"
#include "../util/slam_util.h"
#include "../input/pointcloud_helpers.h"

#include "solver_vis_ros.h"

namespace nautilus::visualization {

using slam_types::SLAMState2D;
using Eigen::Vector2f;

static geometry_msgs::Pose ConstructPoseMsg(double * pose) {
  geometry_msgs::Pose p;
  p.position.x = pose[0];
  p.position.y = pose[1];
  p.orientation.z = sin(pose[2] / 2);
  p.orientation.w = cos(pose[2] / 2);
  return p;
}

SolverVisualizerROS::SolverVisualizerROS(std::shared_ptr<slam_types::SLAMState2D>& state, ros::NodeHandle& n)
  : SolverVisualizer(state) {
  points_pub_ = n.advertise<sensor_msgs::PointCloud2>("/nautilus/all_points", 10);
  poses_pub_ = n.advertise<geometry_msgs::PoseArray>("/nautilus/all_poses", 10);
}

void SolverVisualizerROS::DrawSolution() const {
  // Compile all the points and poses into lists to be published.
  auto solution = state_->solution;
  auto problem = state_->problem;
  std::vector<Vector2f> all_points;
  geometry_msgs::PoseArray all_poses;
  CHECK_EQ(solution.size(), problem.nodes.size());
  for (size_t i = 0; i < solution.size(); i++) {
    auto &pointcloud = problem.nodes[i].lidar_factor.pointcloud;
    if (pointcloud.size() == 0) {
      // Skip empty pointclouds.
      continue;
    }
    // Otherwise, transform all the points by this pose.
    auto transformed_pointcloud = TransformPointcloud(solution[i].pose, pointcloud);
    // Construct an arrow to represent the point.
    auto pose_msg = ConstructPoseMsg(solution[i].pose);
    all_poses.poses.push_back(pose_msg);
    all_points.insert(all_points.end(), transformed_pointcloud.begin(), transformed_pointcloud.end());
  }
  // Publishing the points.
  sensor_msgs::PointCloud2 points_msg;
  InitPointcloud(&points_msg);
  PublishPointcloud(all_points, &points_msg, points_pub_);
  // Publishing the poses.
  all_poses.header.frame_id = "map";
  poses_pub_.publish(all_poses);
}

void SolverVisualizerROS::DrawCorrespondence(const Correspondence &) const {
  // TODO:
}

}  // nautilus::visualization
