//
// Created by jack on 9/30/19.
//

#include "eigen3/Eigen/Dense"

#include "slam_type_builder.h"
#include "pointcloud_helpers.h"
#include "math_util.h"
#include "lidar_slam/CobotOdometryMsg.h"
#include "math_util.h"

using pointcloud_helpers::LaserScanToPointCloud;
using slam_types::LidarFactor;
using slam_types::RobotPose2D;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMProblem2D;
using math_util::AngleDist;
using Eigen::Vector2f;
using Eigen::Rotation2Df;
using lidar_slam::CobotOdometryMsg;

void SLAMTypeBuilder::AddOdomFactor(
        std::vector<OdometryFactor2D>& odom_factors) {
  CHECK_GE(nodes_.size(), 2);
  auto node_i = nodes_[nodes_.size() - 1];
  auto node_j = nodes_[nodes_.size() - 2];
  double angle = node_i.pose.angle - node_j.pose.angle;
  Vector2f translation = node_i.pose.loc - node_j.pose.loc;
  OdometryFactor2D odom_factor(nodes_.size() - 2,
                               nodes_.size() - 1,
                               translation, angle);
  odom_factors.emplace_back(odom_factor);
}

bool SLAMTypeBuilder::Done() {
  return (lidar_callback_count_ >= pose_num_max_);
}

void SLAMTypeBuilder::LidarCallback(sensor_msgs::LaserScan& laser_scan) {
  if (lidar_callback_count_ >= pose_num_max_) {
    return;
  }
  // We only want one odometry between each lidar callback.
  if ((differential_odom_ && diff_tracking.ReadyForLidar()) ||
       odom_tracking.ReadyForLidar()) {
    // Transform this laser scan into a point cloud.s
    double max_range =
      (range_cutoff_ <= 0)? laser_scan.range_max : range_cutoff_;
    std::vector<Vector2f> pointcloud =
      LaserScanToPointCloud(laser_scan, max_range, true);
    LidarFactor lidar_factor(pose_id_, pointcloud);
    RobotPose2D pose;
    if (differential_odom_) {
      pose = diff_tracking.GetPose();
    } else {
      pose = odom_tracking.GetPose();
    }
    SLAMNode2D slam_node(pose_id_, laser_scan.header.stamp.toSec(), pose,
                         lidar_factor);
    nodes_.push_back(slam_node);
    if (pose_id_ > 0) {
      AddOdomFactor(odom_factors_);
    }
    pose_id_++;
    lidar_callback_count_++;
  }
}

float ZRadiansFromQuaterion(geometry_msgs::Quaternion& q) {
  // Protect against case of gimbal lock which will give us a singular transformation.
  // http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
  if ((q.x * q.y) + (q.z * q.w) == 0.5) {
    return 0.0;
  } else if ((q.x * q.y) + (q.z * q.w) == -0.5) {
    return 0.0;
  }
  double first_arg = 2.0f * (q.w * q.z + q.x * q.z);
  double second_arg = 1.0f - (2.0f * (q.y * q.y + q.z * q.z));
  return atan2(first_arg, second_arg);
}

void SLAMTypeBuilder::OdometryCallback(nav_msgs::Odometry& odometry) {
  odom_tracking.OdometryCallback(odometry);
}

void
SLAMTypeBuilder::OdometryCallback(CobotOdometryMsg& odometry) {
  diff_tracking.OdometryCallback(odometry);
}

slam_types::SLAMProblem2D SLAMTypeBuilder::GetSlamProblem() {
    SLAMProblem2D slam_problem(nodes_, odom_factors_);
  return slam_problem;
}

SLAMTypeBuilder::SLAMTypeBuilder(uint64_t pose_num,
                                 bool differential_odom,
                                 double range_cutoff) :
  pose_num_max_(pose_num),
  differential_odom_(differential_odom),
  range_cutoff_(range_cutoff) {}

void DifferentialOdometryTracking::OdometryCallback(
        lidar_slam::CobotOdometryMsg &odometry) {
  if (!odom_initialized_) {
    odom_initialized_ = true;
    pending_rotation_ = 0;
    pending_translation_ = Vector2f(0, 0);
  } else {
    pending_rotation_ = math_util::angle_mod(odometry.dr + pending_rotation_);
    pending_translation_ += Vector2f(odometry.dx, odometry.dy);
  }
}

RobotPose2D DifferentialOdometryTracking::GetPose() {
  // We multiply by the total rotation because this translation
  // is in the context of the last robots position's frame.
  total_translation += Rotation2Df(total_rotation) * pending_translation_;
  total_rotation = math_util::angle_mod(total_rotation + pending_rotation_);
  pending_translation_ = Vector2f(0,0);
  pending_rotation_ = 0.0;
  return RobotPose2D(total_translation, total_rotation);
}

void AbsoluteOdometryTracking::OdometryCallback(nav_msgs::Odometry &odometry) {
  if (!odom_initialized_) {
    init_odom_translation_ = Vector2f(odometry.pose.pose.position.x,
                                      odometry.pose.pose.position.y);
    init_odom_angle_ = ZRadiansFromQuaterion(odometry.pose.pose.orientation);
    last_odom_translation_ = init_odom_translation_;
    last_odom_angle_ = init_odom_angle_;
    odom_initialized_ = true;
  }
  odom_angle_ = ZRadiansFromQuaterion(odometry.pose.pose.orientation);
  odom_translation_ = Vector2f(odometry.pose.pose.position.x,
                               odometry.pose.pose.position.y);
}

RobotPose2D AbsoluteOdometryTracking::GetPose() {
  Vector2f translation = odom_translation_ - init_odom_translation_;
  double angle = odom_angle_;
  last_odom_angle_ = odom_angle_;
  last_odom_translation_ = odom_translation_;
  return RobotPose2D(translation, angle);
}
