// Copyright (c) 2018 joydeepb@cs.umass.edu
// College of Information and Computer Sciences,
// University of Massachusetts Amherst
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef __SLAM_TYPES_H__
#define __SLAM_TYPES_H__

#include <utility>
#include <vector>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

#include "math_util.h"


namespace slam_types {
  
// Pinhole camera intrinsics parameters.
// The convention used here matches that of OpenCV:
// https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/
//     camera_calibration.html
struct CameraIntrinsics {
  // Focal length x.
  float fx;
  // Focal length y.
  float fy;
  // Principal point x.
  float cx;
  // Principal point y.
  float cy;
};

// The camera extrinsics consists of the coordinate transform from the
// camera to the robot pose. That is, it consists of the translation and
// rotation that takes a point from the camera frame to the robot frame.
struct CameraExtrinsics {
  // 3D vector of translation.
  float translation[3];
  // Rotation in scaled angle-axis (Lie algebra) form.
  float rotation[3];
};

struct VisionFeature {
  // Index of this feature in the features vector for the node, stored here
  // for redundancy.
  uint64_t feature_idx;
  // Camera pixel location of feature.
  Eigen::Vector2f pixel;
  // Estimated 3d location in the camera frame.
  Eigen::Vector3f point3d;
  // Default constructor: do nothing.
  VisionFeature() {}
  // Convenience constructor: initialize everything.
  VisionFeature(uint64_t idx,
                const Eigen::Vector2f& p,
                const Eigen::Vector3f& point3d) :
      feature_idx(idx), pixel(p), point3d(point3d) {}
};

struct FeatureMatch {
  // Feature ID from the initial pose.
  uint64_t feature_idx_initial;
  // Feature ID from current pose.
  uint64_t feature_idx_current;
  // Default constructor: do nothing.
  FeatureMatch() {}
  // Convenience constructor: initialize everything.
  FeatureMatch(uint64_t fid_initial,
               uint64_t fid_current) :
    feature_idx_initial(fid_initial),
    feature_idx_current(fid_current) {}
};

struct VisionFactor {
  // ID of the pose where the features were *first* observed.
  uint64_t pose_idx_initial;
  // ID of second pose.
  uint64_t pose_idx_current;
  // Pair of feature ID from first pose, and feature ID from second pose,
  // and feature ID from initial pose.
  std::vector<FeatureMatch> feature_matches;
  // Default constructor: do nothing.
  VisionFactor() {}
  // Convenience constructor: initialize everything.
  VisionFactor(
      uint64_t pose_initial,
      uint64_t pose_current,
      const std::vector<slam_types::FeatureMatch>& feature_matches) :
      pose_idx_initial(pose_initial), pose_idx_current(pose_current),
      feature_matches(feature_matches) {}
};

struct LidarFactor {
    // IDs of the poses
    uint64_t pose_id;
    std::vector<Eigen::Vector2f> pointcloud;
    LidarFactor() {
      pose_id = 0;
    }
    LidarFactor(uint64_t pose_id,
                std::vector<Eigen::Vector2f>& pointcloud) :
                pose_id(pose_id),
                pointcloud(pointcloud) {}
};

struct RobotPose {
  // Robot location.
  Eigen::Vector3f loc;
  // Robot angle: rotates points from robot frame to global.
  Eigen::Quaternionf angle;
  // Default constructor: do nothing.
  RobotPose() {}
  // Convenience constructor: initialize everything.
  RobotPose(const Eigen::Vector3f& loc,
            const Eigen::Quaternionf& angle) :
      loc(loc), angle(angle) {}
  // Return a transform from the robot to the world frame for this pose.
  Eigen::Affine3f RobotToWorldTf() const {
    return (Eigen::Translation3f(loc) * angle);
  }
  // Return a transform from the world to the robot frame for this pose.
  Eigen::Affine3f WorldToRobotTf() const {
    return ((Eigen::Translation3f(loc) * angle).inverse());
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct RobotPose2D {
    // Robot location.
    Eigen::Vector2f loc;
    // Robot angle: rotates points from robot frame to global.
    // RADIANS
    float angle;
    // Default constructor: do nothing.
    RobotPose2D() {}
    // Convenience constructor: initialize everything.
    RobotPose2D(const Eigen::Vector2f&  loc,
                const float angle) :
                loc(loc), angle(angle) {}
    // Return a transform from the robot to the world frame for this pose.
    Eigen::Affine2f RobotToWorldTf() const {
      return (Eigen::Translation2f(loc) *
              Eigen::Rotation2D<float>(angle)
                  .toRotationMatrix());
    }
    // Return a transform from the world to the robot frame for this pose.
    Eigen::Affine2f WorldToRobotTf() const {
      return ((Eigen::Translation2f(loc) *
               Eigen::Rotation2D<float>(angle)
                   .toRotationMatrix()).inverse());
    }
};

// Represents an IMU or odometry, or fused IMU-odometry observation.
struct OdometryFactor {
  // ID of first pose.
  uint64_t pose_i;
  // ID of second pose.
  uint64_t pose_j;
  // Translation to go from pose i to pose j.
  Eigen::Vector3f translation;
  // Rotation to go from pose i to pose j.
  Eigen::Quaternionf rotation;
  // Default constructor: do nothing.
  OdometryFactor() {
    pose_i = 0;
    pose_j = 0;
  }
  // Convenience constructor: initialize everything.
  OdometryFactor(uint64_t pose_i,
                         uint64_t pose_j,
                         Eigen::Vector3f& translation,
                         Eigen::Quaternionf& rotation) :
      pose_i(pose_i), pose_j(pose_j), translation(translation),
      rotation(rotation) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct OdometryFactor2D {
    // ID of first pose.
    uint64_t pose_i{};
    // ID of second pose.
    uint64_t pose_j{};
    // Translation to go from pose i to pose j.
    Eigen::Vector2f translation;
    // Rotation to go from pose i to pose j.
    float rotation{};
    // Default constructor: do nothing.
    OdometryFactor2D() = default;
    // Convenience constructor: initialize everything.
    OdometryFactor2D(uint64_t pose_i,
                     uint64_t pose_j,
                     Eigen::Vector2f& translation,
                     float rotation) :
            pose_i(pose_i), pose_j(pose_j), translation(translation),
            rotation(rotation) {}
};


struct SLAMNode {
  // Pose index for this node in the nodes vector for the slam problem.
  uint64_t node_idx{};
  // Timestamp.
  double timestamp{};
  // 6DOF parameters.
  RobotPose pose;
  // Observed vision features.
  std::vector<VisionFeature> features;
  // Default constructor: do nothing.
  SLAMNode() = default;
  // Convenience constructor, initialize all components.
  SLAMNode(uint64_t idx,
           double timestamp,
           const RobotPose& pose,
           const std::vector<VisionFeature>& features) :
      node_idx(idx), timestamp(timestamp), pose(pose), features(features) {}
};

struct SLAMNode2D {
    // Pose index for this node in the nodes vector for the slam problem.
    uint64_t node_idx{};
    // Timestamp.
    double timestamp{};
    // 3DOF parameters.
    RobotPose2D pose;
    // Observed Lidar Factor
    LidarFactor lidar_factor;
    // Default constructor: do nothing.
    SLAMNode2D() = default;
    // Convenience constructor, initialize all components.
    SLAMNode2D(uint64_t idx,
             double timestamp,
             const RobotPose2D& pose,
             const LidarFactor& lidar_factor) :
            node_idx(idx),
            timestamp(timestamp),
            pose(pose),
            lidar_factor(lidar_factor) {}
};

struct SLAMProblem {
    // Nodes in the pose graph.
    std::vector<SLAMNode> nodes;
    // Vision correspondences.
    std::vector<VisionFactor> vision_factors;
    // Odometry / IMU correspondences.
    std::vector<OdometryFactor> odometry_factors;
    // Default constructor, do nothing.
    SLAMProblem() = default;
    // Convenience constructor for initialization.
    SLAMProblem(const std::vector<SLAMNode>& nodes,
                const std::vector<VisionFactor>& vision_factors,
                const std::vector<OdometryFactor>& odometry_factors) :
            nodes(nodes),
            vision_factors(vision_factors),
            odometry_factors(odometry_factors) {}
};

struct SLAMProblem2D {
    // Nodes in the pose graph.
    std::vector<SLAMNode2D> nodes;
    // Odometry / IMU correspondences.
    std::vector<OdometryFactor2D> odometry_factors;
    // Default constructor, do nothing.
    SLAMProblem2D() = default;
    // Convenience constructor for initialization.
    SLAMProblem2D(std::vector<SLAMNode2D>  nodes,
                  std::vector<OdometryFactor2D>  odometry_factors) :
            nodes(std::move(nodes)),
            odometry_factors(std::move(odometry_factors)){}
};

struct SLAMNodeSolution {
  // Pose ID.
  uint64_t node_idx;
  // Timestamp.
  double timestamp;
  // 6DOF parameters: tx, ty, tx, angle_x, angle_y, angle_z. Note that
  // angle_* are the coordinates in scaled angle-axis form.
  double pose[6];
  // Observed vision feature inverse depths.
  std::vector<double> inverse_depths;
  // Corresponding flag of whether the point is to be part of the map.
  std::vector<bool> point_in_map;
  // Convenience constructor, initialize all values.
  explicit SLAMNodeSolution(const SLAMNode& n) :
      node_idx(n.node_idx),
      timestamp(n.timestamp),
      pose{0, 0, 0, 0, 0, 0},
      inverse_depths(n.features.size(), 1.0),
      point_in_map(n.features.size(), false) {
    const Eigen::AngleAxisf rotation(n.pose.angle);
    const Eigen::Vector3f rotation_aa = rotation.angle() * rotation.axis();
    pose[0] = n.pose.loc.x();
    pose[1] = n.pose.loc.y();
    pose[2] = n.pose.loc.z();
    pose[3] = rotation_aa.x();
    pose[4] = rotation_aa.y();
    pose[5] = rotation_aa.z()+0.02;
  }
};

struct SLAMNodeSolution2D {
    // Pose ID.
    uint64_t node_idx{};
    // Timestamp.
    double timestamp{};
    // 3DOF parameters: tx, ty, angle. Note that
    // angle_* are the coordinates in scaled angle-axis form.
    double pose[3]{};
    // Convenience constructor, initialize all values.
    explicit SLAMNodeSolution2D(const SLAMNode2D& n) :
                       node_idx(n.node_idx),
                       timestamp(n.timestamp),
                       pose{n.pose.loc.x(), n.pose.loc.y(), n.pose.angle} {}

    SLAMNodeSolution2D() = default;
};

}  // namespace slam_types

#endif  // __SLAM_TYPES_H__
