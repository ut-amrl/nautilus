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
  CHECK_EQ(odom_initialized_, true);
  if (!differential_odom_) {
    Eigen::Matrix2f last_rot_mat =
        Eigen::Rotation2D<float>(last_odom_angle_)
            .toRotationMatrix();
    Vector2f translation =
        last_rot_mat.inverse() * (odom_translation_ - last_odom_translation_);
    Eigen::Matrix2f curr_rot_mat =
        Eigen::Rotation2D<float>(odom_angle_)
            .toRotationMatrix();
    Eigen::Matrix2f rotation =
        curr_rot_mat * last_rot_mat.transpose();
    // Recover angle from rotation matrix.
    double angle = atan2(rotation(0, 1), rotation(0, 0));
    odom_factors.emplace_back(pose_id_ - 1, pose_id_, translation, angle);
    last_odom_angle_ = odom_angle_;
    last_odom_translation_ = odom_translation_;
  } else {
    Vector2f translation = odom_translation_;
    float angle = math_util::angle_mod(odom_angle_);
    if (pose_id_ == 1) {
      // Because this is differential, every odometry message after
      // the first will be an accurate change amount,
      // but the first is garbage.
      translation = Vector2f(0, 0);
      angle = 0;
    }
    odom_factors.emplace_back(pose_id_ - 1,
                              pose_id_,
                              translation,
                              angle);
    odom_angle_ = 0;
    odom_translation_ = Vector2f(0,0);
  }
}

bool SLAMTypeBuilder::Done() {
  return (lidar_callback_count_ >= pose_num_max_);
}

void SLAMTypeBuilder::LidarCallback(sensor_msgs::LaserScan& laser_scan) {
  if (lidar_callback_count_ >= pose_num_max_) {
    return;
  }
  // We only want one odometry between each lidar callback.
  if (odom_initialized_ &&
     ((last_odom_translation_ - odom_translation_).norm() > 0.20 ||
     (AngleDist(odom_angle_, last_odom_angle_) > M_PI / 18.0))) {
    // Transform this laser scan into a point cloud.s
    std::vector<Vector2f> pointcloud = LaserScanToPointCloud(laser_scan);
    LidarFactor lidar_factor(pose_id_, pointcloud);
    RobotPose2D pose;
    if (!differential_odom_) {
      pose = RobotPose2D(odom_translation_ - init_odom_translation_,
                         odom_angle_); //TODO: Maybe not good idea to not subtract initial angle.
    } else {
      total_translation += Rotation2Df(total_rotation) * odom_translation_;
      total_rotation = math_util::angle_mod(total_rotation + odom_angle_);
      pose = RobotPose2D(total_translation, total_rotation);
    }
    SLAMNode2D slam_node(pose_id_, laser_scan.header.stamp.toSec(), pose,
                         lidar_factor);
    nodes_.push_back(slam_node);
    CHECK_EQ(nodes_.size(), pose_id_ + 1);
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

void
SLAMTypeBuilder::DifferentialOdometryCallback(CobotOdometryMsg& odometry) {
  if (!odom_initialized_) {
    odom_initialized_ = true;
    last_odom_angle_ = 0;
    last_odom_translation_ = Vector2f(0, 0);
    odom_angle_ = 0;
    odom_translation_ = Vector2f(0, 0);
  } else {
    odom_angle_ = math_util::angle_mod(odometry.dr + odom_angle_);
    odom_translation_ += Vector2f(odometry.dx, odometry.dy);
  }
}

slam_types::SLAMProblem2D SLAMTypeBuilder::GetSlamProblem() {
    SLAMProblem2D slam_problem(nodes_, odom_factors_);
  return slam_problem;
}

SLAMTypeBuilder::SLAMTypeBuilder(uint64_t pose_num, bool differential_odom) :
  pose_num_max_(pose_num),
  differential_odom_(differential_odom) {}
