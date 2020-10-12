//
// Created by jack on 9/15/19.
//

#ifndef SRC_POINTCLOUD_HELPERS_H_
#define SRC_POINTCLOUD_HELPERS_H_

#include <sensor_msgs/LaserScan.h>

#include <vector>

#include "eigen3/Eigen/Dense"
#include "ros/package.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"

namespace nautilus {
void InitPointcloud(sensor_msgs::PointCloud2* point);
void PushBackBytes(float val, sensor_msgs::PointCloud2* ptr);
void PublishPointcloud(const std::vector<Eigen::Vector2f>& points,
                       sensor_msgs::PointCloud2* point_cloud,
                       ros::Publisher& pub);
std::vector<Eigen::Vector2f> normalizePointCloud(
    const std::vector<Eigen::Vector2f>& pointcloud, double range);
sensor_msgs::PointCloud2 EigenPointcloudToRos(
    const std::vector<Eigen::Vector2f>& pointcloud);
std::vector<Eigen::Vector2f> LaserScanToPointCloud(
    const sensor_msgs::LaserScan& laser_scan, double max_range);

};      // namespace nautilus
#endif  // SRC_POINTCLOUD_HELPERS_H_
