//
// Created by jack on 9/30/19.
//

#include "eigen3/Eigen/Dense"

#include "slam_type_builder.h"
#include "pointcloud_helpers.h"

using pointcloud_helpers::LaserScanToPointCloud;
using slam_types::LidarFactor;
using slam_types::RobotPose2D;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMProblem2D;

void SLAMTypeBuilder::AddOdomFactor(
        std::vector<OdometryFactor2D>& odom_factors) {
  CHECK_EQ(odom_initialized_, true);
  Eigen::Matrix2f last_rot_mat =
      Eigen::Rotation2D<float>(last_odom_angle_)
          .toRotationMatrix();
  Eigen::Vector2f translation =
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
}

void SLAMTypeBuilder::LidarCallback(sensor_msgs::LaserScan& laser_scan) {
  if (lidar_callback_count >= LIDAR_CALLBACK_CUTOFF) {
    return;
  }
  // We only want one odometry between each lidar callback.
  if (odom_initialized_ && ((last_odom_translation_ - odom_translation_).norm() > 0.1 || (abs(odom_angle_ - last_odom_angle_) > M_PI / 4))) {
    // Transform this laser scan into a point cloud.s
    std::vector<Eigen::Vector2f> pointcloud = LaserScanToPointCloud(laser_scan);
    LidarFactor lidar_factor(pose_id_, pointcloud);
    RobotPose2D pose(odom_translation_ - init_odom_translation_,
                     odom_angle_); //TODO: Maybe not good idea to not subtract initial angle.
    SLAMNode2D slam_node(pose_id_, laser_scan.header.stamp.toSec(), pose,
                         lidar_factor);
    nodes_.push_back(slam_node);
    if (pose_id_ > 0) {
      AddOdomFactor(odom_factors_);
    }
    pose_id_++;
    lidar_callback_count++;
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
    init_odom_translation_ = Eigen::Vector2f(odometry.pose.pose.position.x,
                                             odometry.pose.pose.position.y);
    init_odom_angle_ = ZRadiansFromQuaterion(odometry.pose.pose.orientation);
    last_odom_translation_ = init_odom_translation_;
    last_odom_angle_ = init_odom_angle_;
    odom_initialized_ = true;
    printf("Initial Angle in Rad: %lf\n", init_odom_angle_);
  }
  odom_angle_ = ZRadiansFromQuaterion(odometry.pose.pose.orientation);
  odom_translation_ = Eigen::Vector2f(odometry.pose.pose.position.x,
                                      odometry.pose.pose.position.y);
}

slam_types::SLAMProblem2D SLAMTypeBuilder::GetSlamProblem() {
  SLAMProblem2D slam_problem(nodes_, odom_factors_);
  return slam_problem;
}
