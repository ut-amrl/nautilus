#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>

#include "../util/slam_types.h"
#include "Eigen/Dense"
#include "ceres/ceres.h"

namespace nautilus {

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
  std::vector<Eigen::Vector2f> points_on_feature;
  LCPose(uint64_t node_idx, std::vector<Eigen::Vector2f> points_on_feature)
      : node_idx(node_idx), points_on_feature(points_on_feature) {}
};

struct HitlLCConstraint {
  std::vector<LCPose> line_a_poses;
  std::vector<LCPose> line_b_poses;
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
  std::vector<Eigen::Vector2f> source_points;
  std::vector<Eigen::Vector2f> target_points;
  std::vector<Eigen::Vector2f> source_normals;
  std::vector<Eigen::Vector2f> target_normals;
  double* source_pose;
  double* target_pose;
  uint64_t source_index;
  uint64_t target_index;
  double difference = 0.0;
  PointCorrespondences(double* source_pose, double* target_pose,
                       uint64_t source_index, uint64_t target_index)
      : source_pose(source_pose),
        target_pose(target_pose),
        source_index(source_index),
        target_index(target_index) {
    CHECK_NOTNULL(source_pose);
    CHECK_NOTNULL(target_pose);
  }
  PointCorrespondences(uint64_t source_index, uint64_t target_index)
      : source_index(source_index), target_index(target_index) {}

  PointCorrespondences()
      : source_pose(nullptr),
        target_pose(nullptr),
        source_index(0),
        target_index(0) {}

  void AddCorrespondence(Eigen::Vector2f source_point,
                         Eigen::Vector2f source_normal,
                         Eigen::Vector2f target_point,
                         Eigen::Vector2f target_normal) {
    source_points.push_back(source_point);
    source_normals.push_back(source_normal);
    target_points.push_back(target_point);
    target_normals.push_back(target_normal);
    difference += (source_point - target_point).norm();
  }
};

struct ResidualDesc {
  size_t node_i;
  size_t node_j;
  ceres::ResidualBlockId id;
  ResidualDesc(size_t node_i, size_t node_j, ceres::ResidualBlockId id)
      : node_i(node_i), node_j(node_j), id(id) {}
};

struct CeresInformation {
  void ResetProblem() {
    problem.reset(new ceres::Problem());
    res_descriptors.clear();
    cost_valid = false;
    cost = 0.0;
  }
  bool cost_valid = false;
  double cost = 0.0;
  std::shared_ptr<ceres::Problem> problem;
  std::vector<ResidualDesc> res_descriptors;
};

}  // namespace nautilus

#endif
