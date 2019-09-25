//
// Created by jack on 9/15/19.
//

#ifndef ICP_POINTCLOUD_HELPERS_H
#define ICP_POINTCLOUD_HELPERS_H

#include "ros/package.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "eigen3/Eigen/Dense"

using Eigen::Vector2d;
using sensor_msgs::PointCloud2;
using ros::Publisher;

namespace pointcloud_helpers {
  void InitPointcloud(PointCloud2* point);
  void PushBackBytes(float val, sensor_msgs::PointCloud2& ptr);
  void PublishPointcloud(const std::vector<Vector2d>& points,
                         PointCloud2& point_cloud,
                         Publisher& pub);
};
#endif //ICP_POINTCLOUD_HELPERS_H
