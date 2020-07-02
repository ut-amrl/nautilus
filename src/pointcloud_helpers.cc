//
// Created by jack on 9/15/19.
//

#include "./pointcloud_helpers.h"
#include <glog/logging.h>
#include <numeric>

#include "eigen3/Eigen/Dense"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"

#include "./kdtree.h"
#include "./math_util.h"

using Eigen::Matrix2f;
using Eigen::Rotation2D;
using Eigen::Vector2f;
using math_util::NormalsSimilar;
using std::pair;
using std::vector;

// TODO: Throw out this method for stf filtering.
#define GLANCING_THRESHOLD 0.25

vector<Vector2f> pointcloud_helpers::LaserScanToPointCloud(
    sensor_msgs::LaserScan& laser_scan, double max_range) {
  vector<Vector2f> pointcloud;
  float angle_offset = laser_scan.range_min;
  for (size_t index = 0; index < laser_scan.ranges.size(); index++) {
    float range = laser_scan.ranges[index];
    if (range >= laser_scan.range_min && range <= max_range) {
      // Only accept valid ranges.
      // Then we must rotate the point by the specified angle at that distance.
      Vector2f point(range, 0.0);
      Matrix2f rot_matrix =
          Rotation2D<float>(laser_scan.angle_min +
                            (laser_scan.angle_increment * index))
              .toRotationMatrix();
      point = rot_matrix * point;
      pointcloud.emplace_back(point);
    }
    angle_offset += laser_scan.angle_increment;
  }
  return pointcloud;
}

void pointcloud_helpers::InitPointcloud(PointCloud2* point) {
  std::string arr[3] = {"x", "y", "z"};
  point->header.seq = 1;
  point->header.stamp = ros::Time::now();
  point->header.frame_id = "map";
  sensor_msgs::PointField field;
  int offset = 0;
  field.datatype = 7;
  field.count = 1;
  for (std::string type : arr) {
    field.offset = offset;
    field.name = type;
    point->fields.push_back(field);
    offset += 4;
  }
  point->height = 1;
  point->width = 0;
  point->is_bigendian = false;
  point->point_step = 12;
  point->row_step = 0;
  point->is_dense = true;
}

void pointcloud_helpers::PushBackBytes(float val,
                                       sensor_msgs::PointCloud2& ptr) {
  uint8_t* data_ptr = reinterpret_cast<uint8_t*>(&val);
  for (int i = 0; i < 4; i++) {
    ptr.data.push_back(data_ptr[i]);
  }
}

void pointcloud_helpers::PublishPointcloud(const vector<Vector2f>& points,
                                           PointCloud2& point_cloud,
                                           Publisher& pub) {
  for (uint64_t i = 0; i < points.size(); i++) {
    Vector2f vec = points[i];
    PushBackBytes(vec[0], point_cloud);
    PushBackBytes(vec[1], point_cloud);
    PushBackBytes(0.0f, point_cloud);
  }
  point_cloud.width = points.size();
  pub.publish(point_cloud);
  point_cloud.width = 0;
  point_cloud.data.clear();
}

PointCloud2 pointcloud_helpers::EigenPointcloudToRos(
    const vector<Vector2f>& pointcloud) {
  PointCloud2 point_msg;
  InitPointcloud(&point_msg);
  for (uint64_t i = 0; i < pointcloud.size(); i++) {
    Vector2f vec = pointcloud[i];
    PushBackBytes(vec[0], point_msg);
    PushBackBytes(vec[1], point_msg);
    PushBackBytes(0.0f, point_msg);
  }
  point_msg.height = 1;
  point_msg.width = pointcloud.size();
  return point_msg;
}

std::vector<Vector2f> pointcloud_helpers::normalizePointCloud(
    const vector<Vector2f>& pointcloud, double range) {
  std::vector<Vector2f> normalized(pointcloud.size());
  Vector2f mean = std::accumulate(pointcloud.begin(), pointcloud.end(),
                                  Vector2f(0.0f, 0.0f)) /
                  pointcloud.size();
  for (uint64_t i = 0; i < pointcloud.size(); i++) {
    normalized[i] = (pointcloud[i] - mean) / range;
  }

  return normalized;
}
