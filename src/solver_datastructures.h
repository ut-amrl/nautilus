#pragma once
#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "slam_types.h"

/*----------------------------------------------------------------------------*
 *                            DATA STRUCTURES                                 |
 *----------------------------------------------------------------------------*/

template <typename T>
struct LineSegment {
  const Eigen::Matrix<T, 2, 1> start;
  const Eigen::Matrix<T, 2, 1> end;
  LineSegment(Eigen::Matrix<T, 2, 1>& start, Eigen::Matrix<T, 2, 1>& endpoint)
      : start(start), end(endpoint){};

  LineSegment() {}

  template <typename F>
  LineSegment<F> cast() const {
    typedef Eigen::Matrix<F, 2, 1> Vector2F;
    Vector2F startF = start.template cast<F>();
    Vector2F endF = end.template cast<F>();
    CHECK(ceres::IsFinite(startF.x()));
    CHECK(ceres::IsFinite(startF.y()));
    CHECK(ceres::IsFinite(endF.x()));
    CHECK(ceres::IsFinite(endF.y()));
    return LineSegment<F>(startF, endF);
  }
};

struct LCPose {
  uint64_t node_idx;
  vector<Eigen::Vector2f> points_on_feature;
  LCPose(uint64_t node_idx, vector<Eigen::Vector2f> points_on_feature)
      : node_idx(node_idx), points_on_feature(points_on_feature) {}
};

struct HitlLCConstraint {
  vector<LCPose> line_a_poses;
  vector<LCPose> line_b_poses;
  const LineSegment<float> line_a;
  const LineSegment<float> line_b;
  double chosen_line_pose[3]{0, 0, 0};
  HitlLCConstraint(const LineSegment<float>& line_a,
                   const LineSegment<float>& line_b)
      : line_a(line_a), line_b(line_b) {}
  HitlLCConstraint() {}
};

struct AutoLCConstraint {
  const slam_types::SLAMNode2D* node_a;
  const slam_types::SLAMNode2D* node_b;
  double source_pose[3];
  double target_pose[3];
  float match_ratio;
  Eigen::Vector3f relative_transformation;
};

struct PointCorrespondences {
  vector<Eigen::Vector2f> source_points;
  vector<Eigen::Vector2f> target_points;
  vector<Eigen::Vector2f> source_normals;
  vector<Eigen::Vector2f> target_normals;
  double* source_pose;
  double* target_pose;
  uint64_t source_index;
  uint64_t target_index;
  PointCorrespondences(double* source_pose,
                       double* target_pose,
                       uint64_t source_index,
                       uint64_t target_index)
      : source_pose(source_pose),
        target_pose(target_pose),
        source_index(source_index),
        target_index(target_index) {}
  PointCorrespondences()
      : source_pose(nullptr),
        target_pose(nullptr),
        source_index(0),
        target_index(0) {}
};

struct ResidualDesc {
  size_t node_i;
  size_t node_j;
  ceres::ResidualBlockId id;
  ResidualDesc(size_t node_i, size_t node_j, ceres::ResidualBlockId id)
      : node_i(node_i), node_j(node_j), id(id) {}
};