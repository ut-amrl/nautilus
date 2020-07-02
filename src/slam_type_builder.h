//
// Created by jack on 9/30/19.
//

#ifndef LIDAR_SLAM_SLAM_TYPE_BUILDER_H
#define LIDAR_SLAM_SLAM_TYPE_BUILDER_H

#include "config_reader/config_reader.h"
#include "eigen3/Eigen/Dense"
#include "nautilus/CobotOdometryMsg.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/LaserScan.h"
#include "slam_types.h"

namespace nautilus {

class DifferentialOdometryTracking {
 public:
  DifferentialOdometryTracking() {}

  void OdometryCallback(nautilus::CobotOdometryMsg &odometry);

  slam_types::RobotPose2D GetPose();

  bool ReadyForLidar();

  void ResetInits() {
    total_translation = Eigen::Vector2f(0, 0);
    total_rotation = 0.0f;
  }

 private:
  bool odom_initialized_ = false;
  Eigen::Vector2f pending_translation_ = Eigen::Vector2f(0, 0);
  float pending_rotation_ = 0;
  // This is for calculating the pose in the world frame when
  // using differential odometry.
  Eigen::Vector2f total_translation = Eigen::Vector2f(0, 0);
  float total_rotation = 0.0f;
};

class AbsoluteOdometryTracking {
 public:
  AbsoluteOdometryTracking() {}

  void OdometryCallback(nav_msgs::Odometry &odometry);

  slam_types::RobotPose2D GetPose();

  bool ReadyForLidar();

  void ResetInits() {
    init_odom_angle_ = odom_angle_;
    init_odom_translation_ = odom_translation_;
    pending_translation_ = Eigen::Vector2f(0, 0);
    pending_rotation_ = 0.0f;
    last_odom_angle_ = init_odom_angle_;
    last_odom_translation_ = init_odom_translation_;
  }

 private:
  bool odom_initialized_ = false;
  Eigen::Vector2f init_odom_translation_ = Eigen::Vector2f(0, 0);
  float init_odom_angle_ = 0;
  Eigen::Vector2f odom_translation_ = Eigen::Vector2f(0, 0);
  float odom_angle_ = 0;
  Eigen::Vector2f pending_translation_ = Eigen::Vector2f(0, 0);
  float pending_rotation_ = 0.0;
  // This is the real translation since the last GetPose.
  // It includes the init_odom_translation_ in it.
  Eigen::Vector2f last_odom_translation_ = Eigen::Vector2f(0, 0);
  float last_odom_angle_ = 0;
  // This is the translation in the visualization window. It has been zeroed,
  // and starts from the origin.
  Eigen::Vector2f adjusted_last_translation_ = Eigen::Vector2f(0, 0);
  float adjusted_last_rotation_ = 0.0f;
};

class SLAMTypeBuilder {
 public:
  SLAMTypeBuilder() {}

  void LidarCallback(sensor_msgs::LaserScan &laser_scan);

  void OdometryCallback(nav_msgs::Odometry &odometry);

  void OdometryCallback(CobotOdometryMsg &odometry);

  slam_types::SLAMProblem2D GetSlamProblem();

  bool Done();

 private:
  void AddOdomFactor(std::vector<slam_types::OdometryFactor2D> &);

  uint64_t pose_id_ = 0;
  std::vector<slam_types::SLAMNode2D> nodes_;
  std::vector<slam_types::OdometryFactor2D> odom_factors_;
  // Tracking for different types of odometry.
  AbsoluteOdometryTracking odom_tracking_;
  DifferentialOdometryTracking diff_tracking_;
};

}  // namespace nautilus
#endif  // LIDAR_SLAM_SLAM_TYPE_BUILDER_H
