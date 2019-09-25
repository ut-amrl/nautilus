//
// Created by jack on 9/15/19.
//

#include "pointcloud_helpers.h"

#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"

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

void pointcloud_helpers::PushBackBytes(float val, sensor_msgs::PointCloud2& ptr) {
  uint8_t *data_ptr = reinterpret_cast<uint8_t*>(&val);
  for (int i = 0; i < 4; i++) {
    ptr.data.push_back(data_ptr[i]);
  }
}

void pointcloud_helpers::PublishPointcloud(const std::vector<Vector2d>& points,
                                           PointCloud2& point_cloud,
                                           Publisher& pub) {
  for (uint64_t i = 0; i < points.size(); i++) {
    Vector2d vec = points[i];
    PushBackBytes(float(vec[0]), point_cloud);
    PushBackBytes(float(vec[1]), point_cloud);
    PushBackBytes(0.0f, point_cloud);
  }
  point_cloud.width = points.size();
  pub.publish(point_cloud);
  point_cloud.width = 0;
  point_cloud.data.clear();
}