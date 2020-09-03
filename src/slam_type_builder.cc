//
// Created by jack on 9/30/19.
//

#include "eigen3/Eigen/Dense"

#include "math_util.h"
#include "nautilus/CobotOdometryMsg.h"
#include "pointcloud_helpers.h"
#include "slam_type_builder.h"

using Eigen::Rotation2Df;
using Eigen::Vector2f;
using math_util::AngleDist;
using nautilus::CobotOdometryMsg;
using pointcloud_helpers::LaserScanToPointCloud;
using slam_types::LidarFactor;
using slam_types::OdometryFactor2D;
using slam_types::RobotPose2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMProblem2D;

void SLAMTypeBuilder::AddOdomFactor(
    std::vector<OdometryFactor2D>& odom_factors) {
  CHECK_GE(nodes_.size(), 2);
  auto node_i = nodes_[nodes_.size() - 1];
  auto node_j = nodes_[nodes_.size() - 2];
  double angle = node_i.pose.angle - node_j.pose.angle;
  Vector2f translation = node_i.pose.loc - node_j.pose.loc;
  OdometryFactor2D odom_factor(
      nodes_.size() - 2, nodes_.size() - 1, translation, angle);
  odom_factors.emplace_back(odom_factor);
}

void SLAMTypeBuilder::LidarCallback(sensor_msgs::LaserScan& laser_scan) {
  // We only want one odometry between each lidar callback.
  if (((config::CONFIG_diff_odom && diff_tracking_.ReadyForLidar()) ||
       odom_tracking_.ReadyForLidar()) &&
      !Done()) {
    // Transform this laser scan into a point cloud.s
    double max_range = (config::CONFIG_max_lidar_range <= 0)
                           ? laser_scan.range_max
                           : config::CONFIG_max_lidar_range;

    // TODO wrap in a config-based if
    const size_t truncation_size = 55;
    size_t num_ranges = (laser_scan.angle_max - laser_scan.angle_min) /
                        laser_scan.angle_increment;
    // printf("NUM ranges %ld\n", num_ranges);

    for (size_t i = 0; i < num_ranges; i++) {
      if (i < truncation_size || i > num_ranges - truncation_size) {
        laser_scan.ranges[i] = max_range + 1.0;
      }
    }

    std::vector<Vector2f> pointcloud =
        LaserScanToPointCloud(laser_scan, max_range);
    // laser scan truncation

    LidarFactor lidar_factor(pose_id_, laser_scan, pointcloud);
    RobotPose2D pose;
    // Reset the initial values for everything,
    // we should start at 0 for everything.
    if (pose_id_ == 0) {
      if (config::CONFIG_diff_odom) {
        diff_tracking_.ResetInits();
      } else {
        odom_tracking_.ResetInits();
      }
    }
    if (config::CONFIG_diff_odom) {
      pose = diff_tracking_.GetPose();
    } else {
      pose = odom_tracking_.GetPose();
    }
    SLAMNode2D slam_node(
        pose_id_, laser_scan.header.stamp.toSec(), pose, lidar_factor);
    nodes_.push_back(slam_node);
    if (pose_id_ > 0) {
      AddOdomFactor(odom_factors_);
    }
    pose_id_++;
  }
}

float ZRadiansFromQuaterion(geometry_msgs::Quaternion& q) {
  // Protect against case of gimbal lock which will give us a singular
  // transformation.
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
  odom_tracking_.OdometryCallback(odometry);
}

void SLAMTypeBuilder::OdometryCallback(CobotOdometryMsg& odometry) {
  diff_tracking_.OdometryCallback(odometry);
}

slam_types::SLAMProblem2D SLAMTypeBuilder::GetSlamProblem() {
  SLAMProblem2D slam_problem(nodes_, odom_factors_);
  return slam_problem;
}

size_t SLAMTypeBuilder::GetNodeCount() { return nodes_.size(); }

void DifferentialOdometryTracking::OdometryCallback(
    CobotOdometryMsg& odometry) {
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
  pending_translation_ = Vector2f(0, 0);
  pending_rotation_ = 0.0;
  return RobotPose2D(total_translation, total_rotation);
}

void AbsoluteOdometryTracking::OdometryCallback(nav_msgs::Odometry& odometry) {
  if (!odom_initialized_) {
    init_odom_translation_ =
        Vector2f(odometry.pose.pose.position.x, odometry.pose.pose.position.y);
    init_odom_angle_ = ZRadiansFromQuaterion(odometry.pose.pose.orientation);
    last_odom_translation_ = init_odom_translation_;
    last_odom_angle_ = init_odom_angle_;
    odom_initialized_ = true;
  }
  odom_angle_ = ZRadiansFromQuaterion(odometry.pose.pose.orientation);
  pending_rotation_ = odom_angle_ - last_odom_angle_;
  odom_translation_ =
      Vector2f(odometry.pose.pose.position.x, odometry.pose.pose.position.y);
  pending_translation_ =
      Vector2f(odometry.pose.pose.position.x, odometry.pose.pose.position.y) -
      last_odom_translation_;
}

RobotPose2D AbsoluteOdometryTracking::GetPose() {
  // TODO: Fix the poor starting odometry bug.
  Vector2f total_translation = adjusted_last_translation_;
  float total_rotation = adjusted_last_rotation_;
  // We multiply by the total rotation because this translation
  // is in the context of the last robots position's frame.
  total_translation += Rotation2Df(-init_odom_angle_) * pending_translation_;
  total_rotation = math_util::angle_mod(total_rotation + pending_rotation_);
  pending_translation_ = Vector2f(0, 0);
  pending_rotation_ = 0.0;
  last_odom_angle_ = odom_angle_;
  last_odom_translation_ = odom_translation_;
  adjusted_last_translation_ = total_translation;
  adjusted_last_rotation_ = total_rotation;
  return RobotPose2D(total_translation, total_rotation);
}

bool SLAMTypeBuilder::Done() {
  return pose_id_ >= static_cast<uint64_t>(config::CONFIG_max_pose_num);
}