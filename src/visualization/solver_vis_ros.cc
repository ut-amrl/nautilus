//
// Created by jack on 10/15/20.
//

#include "solver_vis_ros.h"

#include <iostream>
#include <vector>

#include "../input/pointcloud_helpers.h"
#include "../util/gui_helpers.h"
#include "../util/slam_types.h"
#include "../util/slam_util.h"
#include "Eigen/Dense"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseArray.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"

namespace nautilus::visualization {

using Eigen::Vector2f;
using Eigen::Vector3f;
using geometry_msgs::PoseArray;
using slam_types::SLAMState2D;
using std::vector;

static geometry_msgs::Pose ConstructPoseMsg(double* pose) {
  geometry_msgs::Pose p;
  p.position.x = pose[0];
  p.position.y = pose[1];
  p.orientation.z = sin(pose[2] / 2);
  p.orientation.w = cos(pose[2] / 2);
  return p;
}

vector<Eigen::Vector2f> GetAllPoints(std::shared_ptr<SLAMState2D> state) {
  vector<Eigen::Vector2f> all_points;
  CHECK_EQ(state->solution.size(), state->problem.nodes.size());
  for (size_t i = 0; i < state->solution.size(); i++) {
    auto& pointcloud = state->problem.nodes[i].lidar_factor.pointcloud;
    if (pointcloud.size() == 0) {
      // skip empty pointclouds.
      continue;
    }
    // otherwise, transform all the points by this pose.
    auto transformed_pointcloud =
        TransformPointcloud(state->solution[i].pose, pointcloud);
    all_points.insert(all_points.end(), transformed_pointcloud.begin(),
                      transformed_pointcloud.end());
  }
  return all_points;
}

vector<Eigen::Vector2f> GetEdgePoints(std::shared_ptr<SLAMState2D> state) {
  vector<Eigen::Vector2f> edge_points;
  CHECK_EQ(state->solution.size(), state->problem.nodes.size());
  for (size_t i = 0; i < state->solution.size(); i++) {
    auto& pointcloud = state->problem.nodes[i].lidar_factor.edge_points;
    if (pointcloud.size() == 0) {
      // skip empty pointclouds.
      continue;
    }
    // otherwise, transform all the points by this pose.
    auto transformed_pointcloud =
        TransformPointcloud(state->solution[i].pose, pointcloud);
    edge_points.insert(edge_points.end(), transformed_pointcloud.begin(),
                       transformed_pointcloud.end());
  }
  return edge_points;
}

vector<Eigen::Vector2f> GetPlanarPoints(std::shared_ptr<SLAMState2D> state) {
  vector<Eigen::Vector2f> planar_points;
  CHECK_EQ(state->solution.size(), state->problem.nodes.size());
  for (size_t i = 0; i < state->solution.size(); i++) {
    auto& pointcloud = state->problem.nodes[i].lidar_factor.planar_points;
    if (pointcloud.size() == 0) {
      // skip empty pointclouds.
      continue;
    }
    // otherwise, transform all the points by this pose.
    auto transformed_pointcloud =
        TransformPointcloud(state->solution[i].pose, pointcloud);
    planar_points.insert(planar_points.end(), transformed_pointcloud.begin(),
                         transformed_pointcloud.end());
  }
  return planar_points;
}

PoseArray GetAllPoses(std::shared_ptr<SLAMState2D> state) {
  PoseArray all_poses;
  CHECK_EQ(state->solution.size(), state->problem.nodes.size());
  for (size_t i = 0; i < state->solution.size(); i++) {
    auto pose_msg = ConstructPoseMsg(state->solution[i].pose);
    all_poses.poses.push_back(pose_msg);
  }
  return all_poses;
}

void PublishPointcloud(const vector<Eigen::Vector2f>& points,
                       const ros::Publisher& pub) {
  sensor_msgs::PointCloud2 msg;
  InitPointcloud(&msg);
  nautilus::PublishPointcloud(points, &msg, pub);
}

SolverVisualizerROS::SolverVisualizerROS(std::shared_ptr<SLAMState2D> state,
                                         ros::NodeHandle& n)
    : SolverVisualizer(state) {
  points_pub_ =
      n.advertise<sensor_msgs::PointCloud2>("/nautilus/all_points", 10);
  poses_pub_ = n.advertise<geometry_msgs::PoseArray>("/nautilus/all_poses", 10);
  edge_pub_ =
      n.advertise<sensor_msgs::PointCloud2>("/nautilus/edge_points", 10);
  planar_pub_ =
      n.advertise<sensor_msgs::PointCloud2>("/nautilus/planar_points", 10);
  correspondence_pub_ =
      n.advertise<visualization_msgs::Marker>("/nautilus/correspondences", 10);
}

void SolverVisualizerROS::DrawSolution() const {
  // Publish the different point types.
  PublishPointcloud(GetAllPoints(state_), points_pub_);
  PublishPointcloud(GetEdgePoints(state_), edge_pub_);
  PublishPointcloud(GetPlanarPoints(state_), planar_pub_);
  // Publishing the poses.
  auto all_poses = GetAllPoses(state_);
  all_poses.header.frame_id = "map";
  poses_pub_.publish(all_poses);
}

void SolverVisualizerROS::DrawCorrespondence(
    const PointCorrespondences& correspondence) const {
  if (correspondence.source_points.empty() ||
      correspondence.target_points.empty()) {
    return;
  }
  visualization_msgs::Marker line_list_msg;
  gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                gui_helpers::Color4f::kGreen, 0.05f, 0.0f, 0.0f,
                                &line_list_msg);
  auto transformed_a = TransformPointcloud(correspondence.source_pose,
                                           correspondence.source_points);
  auto transformed_b = TransformPointcloud(correspondence.target_pose,
                                           correspondence.target_points);
  CHECK_EQ(transformed_a.size(), transformed_b.size());
  for (size_t i = 0; i < transformed_a.size(); i++) {
    const auto& point_a = transformed_a[i];
    const auto& point_b = transformed_b[i];
    Vector3f point_a_vec3(point_a.x(), point_a.y(), 0.0f);
    Vector3f point_b_vec3(point_b.x(), point_b.y(), 0.0f);
    gui_helpers::AddLine(point_a_vec3, point_b_vec3,
                         gui_helpers::Color4f::kGreen, &line_list_msg);
  }
  correspondence_pub_.publish(line_list_msg);
}

}  // namespace nautilus::visualization
