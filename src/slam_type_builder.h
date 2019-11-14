//
// Created by jack on 9/30/19.
//

#ifndef LIDAR_SLAM_SLAM_TYPE_BUILDER_H
#define LIDAR_SLAM_SLAM_TYPE_BUILDER_H

#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/Odometry.h"
#include "eigen3/Eigen/Dense"
#include "lidar_slam/CobotOdometryMsg.h"

#include "slam_types.h"

#define ROTATION_CHANGE_FOR_LIDAR M_PI / 18
#define TRANSLATION_CHANGE_FOR_LIDAR 0.25

using slam_types::OdometryFactor2D;
using slam_types::RobotPose2D;

class DifferentialOdometryTracking {
 public:
  void OdometryCallback(lidar_slam::CobotOdometryMsg& odometry);
  OdometryFactor2D GetOdomFactor(uint64_t pose_id);
  RobotPose2D GetPose();
  bool ReadyForLidar() {
    return pending_rotation_ >= ROTATION_CHANGE_FOR_LIDAR ||
           pending_translation_.norm() >= TRANSLATION_CHANGE_FOR_LIDAR;
  }
 private:
  bool odom_initialized_ = false;
  Eigen::Vector2f pending_translation_;
  float pending_rotation_;
  // This is for calculating the pose in the world frame when
  // using differential odometry.
  Eigen::Vector2f total_translation = Eigen::Vector2f(0, 0);
  float total_rotation = 0.0f;
};

class AbsoluteOdometryTracking {
 public:
  void OdometryCallback(nav_msgs::Odometry& odometry);
  OdometryFactor2D GetOdomFactor(uint64_t pose_id);
  RobotPose2D GetPose();
  bool ReadyForLidar() {
    float d_angle = math_util::AngleDiff(last_odom_angle_, odom_angle_);
    float d_trans = (odom_translation_ - last_odom_translation_).norm();
    return d_angle >= ROTATION_CHANGE_FOR_LIDAR ||
           d_trans >= TRANSLATION_CHANGE_FOR_LIDAR;
  }
 private:
  bool odom_initialized_ = false;
  Eigen::Vector2f init_odom_translation_;
  float init_odom_angle_;
  Eigen::Vector2f odom_translation_;
  float odom_angle_;
  Eigen::Vector2f last_odom_translation_;
  float last_odom_angle_;
};

class SLAMTypeBuilder {
public:
    SLAMTypeBuilder(uint64_t pose_num, bool differential_odom_);
    void LidarCallback(sensor_msgs::LaserScan& laser_scan);
    void OdometryCallback(nav_msgs::Odometry& odometry);
    void OdometryCallback(lidar_slam::CobotOdometryMsg& odometry);
    slam_types::SLAMProblem2D GetSlamProblem();
    bool Done();
private:
    void AddOdomFactor(std::vector<slam_types::OdometryFactor2D>&);
    uint64_t pose_num_max_ = 0;
    uint64_t pose_id_ = 0;
    std::vector<slam_types::SLAMNode2D> nodes_;
    std::vector<slam_types::OdometryFactor2D> odom_factors_;
    uint64_t lidar_callback_count_ = 0;
    bool differential_odom_ = false;
    // Tracking for different types of odometry.
    AbsoluteOdometryTracking odom_tracking;
    DifferentialOdometryTracking diff_tracking;
};


#endif //LIDAR_SLAM_SLAM_TYPE_BUILDER_H
