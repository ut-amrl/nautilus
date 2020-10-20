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

#include <memory>
#include <utility>
#include <vector>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "sensor_msgs/LaserScan.h"

#include "./kdtree.h"
#include "./math_util.h"
#include "../input/feature_extracter.h"

namespace nautilus {

namespace slam_types {

struct LidarFactor {
  // IDs of the poses
  uint64_t pose_id;
  sensor_msgs::LaserScan scan;
  std::vector<Eigen::Vector2f> pointcloud;
  std::shared_ptr<KDTree<float, 2>> pointcloud_tree;

  std::vector<Eigen::Vector2f> planar_points;
  std::shared_ptr<KDTree<float, 2>> planar_tree;
  std::vector<Eigen::Vector2f> edge_points;
  std::shared_ptr<KDTree<float, 2>> edge_tree;

  LidarFactor() {
    pose_id = 0;
    pointcloud_tree = std::shared_ptr<KDTree<float, 2>>(nullptr);
  }
  LidarFactor(uint64_t pose_id, sensor_msgs::LaserScan& laser_scan,
              std::vector<Eigen::Vector2f>& pointcloud)
      : pose_id(pose_id), pointcloud(pointcloud) {
    KDTree<float, 2>* tree_ptr =
        new KDTree<float, 2>(KDTree<float, 2>::EigenToKD(pointcloud));
    pointcloud_tree = std::shared_ptr<KDTree<float, 2>>(tree_ptr);
    scan = laser_scan;

    // Extract the edge and planar points.
    FeatureExtractor feature_extracter(pointcloud, 0.008, 0.8, 10, 10, 20, 10);
    planar_points = feature_extracter.GetPlanarPoints();
    edge_points = feature_extracter.GetEdgePoints();

    planar_tree =
      std::shared_ptr<KDTree<float, 2>>(new KDTree<float, 2>(KDTree<float, 2>::EigenToKD(planar_points))); 
    edge_tree =
      std::shared_ptr<KDTree<float, 2>>(new KDTree<float, 2>(KDTree<float, 2>::EigenToKD(edge_points)));
  }
};

struct RobotPose2D {
  // Robot location.
  Eigen::Vector2f loc;
  // Robot angle: rotates points from robot frame to global.
  // RADIANS
  float angle{0};
  // Default constructor: do nothing.
  RobotPose2D() : loc(0, 0) {}
  // Convenience constructor: initialize everything.
  RobotPose2D(const Eigen::Vector2f& loc, const float angle)
      : loc(loc), angle(angle) {}
  // Return a transform from the robot to the world frame for this pose.
  Eigen::Affine2f RobotToWorldTf() const {
    return (Eigen::Translation2f(loc) *
            Eigen::Rotation2D<float>(angle).toRotationMatrix());
  }
  // Return a transform from the world to the robot frame for this pose.
  Eigen::Affine2f WorldToRobotTf() const {
    return ((Eigen::Translation2f(loc) *
             Eigen::Rotation2D<float>(angle).toRotationMatrix())
                .inverse());
  }
};

struct OdometryFactor2D {
  // ID of first pose.
  uint64_t pose_i{0};
  // ID of second pose.
  uint64_t pose_j{0};
  // Translation to go from pose i to pose j.
  Eigen::Vector2f translation;
  // Rotation to go from pose i to pose j.
  float rotation{};
  // Default constructor: do nothing.
  OdometryFactor2D() = default;
  // Convenience constructor: initialize everything.
  OdometryFactor2D(uint64_t pose_i, uint64_t pose_j,
                   Eigen::Vector2f& translation, float rotation)
      : pose_i(pose_i),
        pose_j(pose_j),
        translation(translation),
        rotation(rotation) {}
};

struct SLAMNode2D {
  // Pose index for this node in the nodes vector for the slam problem.
  uint64_t node_idx{};
  // Is keyframe?
  bool is_keyframe;
  // Timestamp.
  double timestamp{};
  // 3DOF parameters.
  RobotPose2D pose;
  // Observed Lidar Factor
  LidarFactor lidar_factor;
  // Default constructor: do nothing.
  SLAMNode2D() = default;
  // Convenience constructor, initialize all components.
  SLAMNode2D(uint64_t idx, double timestamp, const RobotPose2D& pose,
             const LidarFactor& lidar_factor)
      : node_idx(idx),
        is_keyframe(false),
        timestamp(timestamp),
        pose(pose),
        lidar_factor(lidar_factor) {}
};

struct SLAMProblem2D {
  // Nodes in the pose graph.
  std::vector<SLAMNode2D> nodes;
  // Odometry / IMU correspondences.
  std::vector<OdometryFactor2D> odometry_factors;
  // Default constructor, do nothing.
  SLAMProblem2D() = default;
  // Convenience constructor for initialization.
  SLAMProblem2D(std::vector<SLAMNode2D> nodes,
                std::vector<OdometryFactor2D> odometry_factors)
      : nodes(std::move(nodes)),
        odometry_factors(std::move(odometry_factors)) {}
};

struct SLAMNodeSolution2D {
  // Pose ID.
  uint64_t node_idx{};
  // Is Keyframe
  bool is_keyframe;

  // Timestamp.
  double timestamp{};
  // 3DOF parameters: tx, ty, angle. Note that
  // angle_* are the coordinates in scaled angle-axis form.
  double pose[3]{0, 0, 0};
  // Convenience constructor, initialize all values.
  explicit SLAMNodeSolution2D(const SLAMNode2D& n)
      : node_idx(n.node_idx),
        is_keyframe(n.is_keyframe),
        timestamp(n.timestamp),
        pose{n.pose.loc.x(), n.pose.loc.y(), n.pose.angle} {}

  SLAMNodeSolution2D() = default;
};

struct SLAMState2D {
  SLAMProblem2D problem;
  std::vector<SLAMNodeSolution2D> solution;

  SLAMState2D(const SLAMProblem2D& prob) : problem(prob) {
    solution.clear();
    for (const auto& node : problem.nodes) {
      solution.emplace_back(node);
    }
  };
};

}  // namespace slam_types
}  // namespace nautilus

#endif  // __SLAM_TYPES_H__
