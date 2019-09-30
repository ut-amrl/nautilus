//
// Created by jack on 9/30/19.
//

#ifndef LIDAR_SLAM_SLAM_TYPE_BUILDER_H
#define LIDAR_SLAM_SLAM_TYPE_BUILDER_H

#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/Odometry.h";
#include "slam_types.h";
#include "Eigen/Dense"

class SLAMTypeBuilder {
public:
    void LidarCallback(sensor_msgs::LaserScan& laser_scan);
    void OdometryCallback(nav_msgs::Odometry& odometry);
    slam_types::SLAMProblem2D GetSlamProblem();
private:
    uint64_t pose_id_ = 0;
    bool odom_initialized_ = false;
    Eigen::Translation2f init_odom_translation_;
    double init_odom_angle_;
    Eigen::Translation2f odom_translation_;
    double odom_angle_;
    std::vector<slam_types::SLAMNode2D> nodes_;
    std::vector<slam_types::OdometryFactor2D> odom_factors_;
};


#endif //LIDAR_SLAM_SLAM_TYPE_BUILDER_H
