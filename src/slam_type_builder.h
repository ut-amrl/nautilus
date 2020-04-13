//
// Created by jack on 9/30/19.
//

#ifndef LIDAR_SLAM_SLAM_TYPE_BUILDER_H
#define LIDAR_SLAM_SLAM_TYPE_BUILDER_H

#include "eigen3/Eigen/Dense"
#include "lidar_slam/CobotOdometryMsg.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/LaserScan.h"

#include "config_reader/config_reader.h"
#include "slam_types.h"

using Eigen::Vector2f;
using slam_types::OdometryFactor2D;
using slam_types::RobotPose2D;

struct SlamTypeBuilderConfig {
  CONFIG_DOUBLE(max_lidar_range, "max_lidar_range");
  CONFIG_BOOL(diff_odom, "differential_odom");
  CONFIG_DOUBLE(max_pose_num, "pose_number");
  CONFIG_DOUBLE(rotation_change, "rotation_change_for_lidar");
  CONFIG_DOUBLE(translation_change, "translation_change_for_lidar");

  SlamTypeBuilderConfig() { config_reader::WaitForInit(); }
};

class DifferentialOdometryTracking {
 public:
  DifferentialOdometryTracking(SlamTypeBuilderConfig config)
      : config_(config) {}
  void OdometryCallback(lidar_slam::CobotOdometryMsg& odometry);
  OdometryFactor2D GetOdomFactor(uint64_t pose_id);
  RobotPose2D GetPose();
  bool ReadyForLidar() {
    return pending_rotation_ >= config_.CONFIG_rotation_change ||
           pending_translation_.norm() >= config_.CONFIG_translation_change;
  }
  void ResetInits() {
    total_translation = Vector2f(0, 0);
    total_rotation = 0.0f;
  }

 private:
  SlamTypeBuilderConfig config_;
  bool odom_initialized_ = false;
  Eigen::Vector2f pending_translation_ = Vector2f(0, 0);
  float pending_rotation_ = 0;
  // This is for calculating the pose in the world frame when
  // using differential odometry.
  Eigen::Vector2f total_translation = Vector2f(0, 0);
  float total_rotation = 0.0f;
};

class AbsoluteOdometryTracking {
 public:
  AbsoluteOdometryTracking(SlamTypeBuilderConfig config) : config_(config) {}
  void OdometryCallback(nav_msgs::Odometry& odometry);
  OdometryFactor2D GetOdomFactor(uint64_t pose_id);
  RobotPose2D GetPose();
  bool ReadyForLidar() {
    float d_angle = math_util::AngleDiff(last_odom_angle_, odom_angle_);
    float d_trans = (odom_translation_ - last_odom_translation_).norm();
    return d_angle >= config_.CONFIG_rotation_change ||
           d_trans >= config_.CONFIG_translation_change;
  }
  void ResetInits() {
    init_odom_angle_ = odom_angle_;
    init_odom_translation_ = odom_translation_;
  }

 private:
  SlamTypeBuilderConfig config_;
  bool odom_initialized_ = false;
  Eigen::Vector2f init_odom_translation_ = Vector2f(0, 0);
  float init_odom_angle_ = 0;
  Eigen::Vector2f odom_translation_ = Vector2f(0, 0);
  float odom_angle_ = 0;
  Eigen::Vector2f last_odom_translation_ = Vector2f(0, 0);
  float last_odom_angle_ = 0;
};

class SLAMTypeBuilder {
 public:
  SLAMTypeBuilder() : odom_tracking_(config_), diff_tracking_(config_) {}
  void LidarCallback(sensor_msgs::LaserScan& laser_scan);
  void OdometryCallback(nav_msgs::Odometry& odometry);
  void OdometryCallback(lidar_slam::CobotOdometryMsg& odometry);
  slam_types::SLAMProblem2D GetSlamProblem();
  bool Done();

 private:
  void AddOdomFactor(std::vector<slam_types::OdometryFactor2D>&);
  uint64_t pose_id_ = 0;
  std::vector<slam_types::SLAMNode2D> nodes_;
  std::vector<slam_types::OdometryFactor2D> odom_factors_;
  SlamTypeBuilderConfig config_;
  // Tracking for different types of odometry.
  AbsoluteOdometryTracking odom_tracking_;
  DifferentialOdometryTracking diff_tracking_;
};

#endif  // LIDAR_SLAM_SLAM_TYPE_BUILDER_H
