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

class SLAMTypeBuilder {
public:
    SLAMTypeBuilder(uint64_t pose_num, bool differential_odom_);
    void LidarCallback(sensor_msgs::LaserScan& laser_scan);
    void OdometryCallback(nav_msgs::Odometry& odometry);
    void DifferentialOdometryCallback(lidar_slam::CobotOdometryMsg& odometry);
    slam_types::SLAMProblem2D GetSlamProblem();
    bool Done();
private:
    uint64_t pose_num_max_ = 0;
    uint64_t pose_id_ = 0;
    bool odom_initialized_ = false;
    Eigen::Vector2f init_odom_translation_;
    float init_odom_angle_;
    Eigen::Vector2f odom_translation_;
    float odom_angle_;
    Eigen::Vector2f last_odom_translation_;
    float last_odom_angle_;
    std::vector<slam_types::SLAMNode2D> nodes_;
    std::vector<slam_types::OdometryFactor2D> odom_factors_;
    void AddOdomFactor(std::vector<slam_types::OdometryFactor2D>&);
    uint64_t lidar_callback_count_ = 0;
    bool differential_odom_ = false;
    // This is for calculating the pose in the world frame when using differential odometry.
    Eigen::Vector2f total_translation = Eigen::Vector2f(0, 0);
    float total_rotation = 0.0f;
};


#endif //LIDAR_SLAM_SLAM_TYPE_BUILDER_H
