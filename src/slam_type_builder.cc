//
// Created by jack on 9/30/19.
//

#include "eigen3/Eigen/Dense"

#include "slam_type_builder.h"
#include "pointcloud_helpers.h"

using pointcloud_helpers::LaserScanToPointCloud;
using slam_types::LidarFactor;
using slam_types::RobotPose2D;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMProblem2D;

void SLAMTypeBuilder::AddOdomFactor(
        std::vector<OdometryFactor2D>* odom_factors) {
  CHECK_EQ(odom_initialized_, true);
  const Eigen::Matrix2f last_rot_mat =
      Eigen::Rotation2D<float>(last_odom_angle_)
          .toRotationMatrix();
  const Eigen::Vector2f translation =
      (odom_translation_ - last_odom_translation_);
  const Eigen::Matrix2f curr_rot_mat =
      Eigen::Rotation2D<float>(odom_angle_)
          .toRotationMatrix();
  const Eigen::Matrix2f rotation =
      last_rot_mat.transpose() * curr_rot_mat;
  // Recover angle from rotation matrix.
  const double angle = atan2(rotation(0, 1), rotation(0, 0));
  odom_factors->emplace_back(pose_id_, pose_id_ + 1, translation, angle);
}

void SLAMTypeBuilder::LidarCallback(sensor_msgs::LaserScan& laser_scan) {
  // We only want one odometry between each lidar callback.
  if (odom_initialized_) {
    // Transform this laser scan into a point cloud.s
    std::vector<Eigen::Vector2f> pointcloud = LaserScanToPointCloud(laser_scan);
    LidarFactor lidar_factor(pose_id_, pointcloud);
    RobotPose2D pose(odom_translation_ - init_odom_translation_,
                     odom_angle_ - init_odom_angle_);
    SLAMNode2D slam_node(pose_id_, laser_scan.header.stamp.toSec(), pose,
                         lidar_factor);
    nodes_.push_back(slam_node);
    AddOdomFactor(&odom_factors_);
    pose_id_ += 1;
  }
}

void SLAMTypeBuilder::OdometryCallback(nav_msgs::Odometry& odometry) {
  if (!odom_initialized_) {
    init_odom_translation_ = Eigen::Vector2f(odometry.pose.pose.position.x,
                                             odometry.pose.pose.position.y);
    init_odom_angle_ = odometry.pose.pose.orientation.y;
    odom_initialized_ = true;
    last_odom_translation_ = init_odom_translation_;
    last_odom_angle_ = init_odom_angle_;
  } else {
    last_odom_translation_ = odom_translation_;
    last_odom_angle_ = odom_angle_;
  }
  odom_angle_ = asin(odometry.pose.pose.orientation.z) * 2;
  odom_translation_ = Eigen::Vector2f(odometry.pose.pose.position.x,
                                      odometry.pose.pose.position.y);
}

slam_types::SLAMProblem2D SLAMTypeBuilder::GetSlamProblem() {
  SLAMProblem2D slam_problem(nodes_, odom_factors_);
  return slam_problem;
}
