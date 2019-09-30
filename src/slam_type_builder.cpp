//
// Created by jack on 9/30/19.
//

#include "slam_type_builder.h"
#include "pointcloud_helpers.h"

using pointcloud_helpers::LaserScanToPointCloud;
using slam_types::LidarFactor;
using slam_types::RobotPose2D;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMProblem2D;

void SLAMTypeBuilder::LidarCallback(sensor_msgs::LaserScan& laser_scan) {
  // Transform this laser scan into a point cloud.
  std::vector<Eigen::Vector2f> pointcloud = LaserScanToPointCloud(laser_scan);
  LidarFactor lidar_factor(pose_id_, pointcloud);
  RobotPose2D pose(odom_translation_ - init_odom_translation_,
                   odom_angle_ - init_odom_angle_);
  SLAMNode2D slam_node(pose_id_, laser_scan.header.stamp, pose, lidar_factor);
  nodes_.push_back(slam_node);
  pose_id_ += 1;
}

void SLAMTypeBuilder::OdometryCallback(nav_msgs::Odometry& odometry) {
  if (!odom_initialized_) {
    init_odom_translation_ = Eigen::Vector2f(odometry.pose.pose.position.x,
                                             odometry.pose.pose.position.y);
    init_odom_angle_ = odometry.pose.pose.orientation.y;
  }
  odom_angle_ = odometry.pose.pose.orientation.y;
  odom_translation_ = Eigen::Vector2f(odometry.pose.pose.position.x,
                                      odometry.pose.pose.position.y);
  odom_factors_.push_back(slam_types::OdometryFactor2D(pose_id_,
                                                       pose_id_ + 1,
                                                       odom_translation_,
                                                       odom_angle_));
  pose_id_ += 1;
}

slam_types::SLAMProblem2D SLAMTypeBuilder::GetSlamProblem() {
  SLAMProblem2D slam_problem(nodes_, odom_factors_);
  return slam_problem;
}