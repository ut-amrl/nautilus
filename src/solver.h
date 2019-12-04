//
// Created by jack on 9/25/19.
//

#ifndef SRC_SOLVER_H_
#define SRC_SOLVER_H_

#include <vector>

#include "glog/logging.h"
#include <ros/node_handle.h>
#include "ros/package.h"
#include "eigen3/Eigen/Dense"
#include "ceres/ceres.h"

#include "./kdtree.h"
#include "./slam_types.h"
#include "lidar_slam/HitlSlamInputMsg.h"

using std::vector;
using Eigen::Vector2f;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using slam_types::SLAMNode2D;
using lidar_slam::HitlSlamInputMsgConstPtr;
using lidar_slam::HitlSlamInputMsg;

template <typename T>
struct LineSegment {
    Eigen::Matrix<T, 2, 1> start;
    Eigen::Matrix<T, 2, 1> endpoint;
    LineSegment(Eigen::Matrix<T, 2, 1>& start,
                Eigen::Matrix<T, 2, 1>& endpoint) :
      start(start), endpoint(endpoint) {};
    template <typename F>
    LineSegment<F> cast() const {
      typedef Eigen::Matrix<F, 2, 1> Vector2F;
      Vector2F startF = start.template cast<F>();
      Vector2F endF = endpoint.template cast<F>();
      CHECK(ceres::IsFinite(startF.x()));
      CHECK(ceres::IsFinite(startF.y()));
      CHECK(ceres::IsFinite(endF.x()));
      CHECK(ceres::IsFinite(endF.y()));
      return LineSegment<F>(startF, endF);
    }
};

#define EPSILON_LINE_ERROR 0.01

template <typename T>
T DistanceToLineSegmentSquared(const Eigen::Matrix<T, 2, 1>& point,
                               const LineSegment<T>& line_seg) {
  typedef Eigen::Matrix<T, 2, 1> Vector2T;
  // Line segment is parametric, with a start point and endpoint.
  // Parameterized by t between 0 and 1.
  // We can get the point on the line by projecting the start -> point onto
  // this line.
  Eigen::Hyperplane<T, 2> line =
          Eigen::Hyperplane<T, 2>::Through(line_seg.start, line_seg.endpoint);
  Eigen::Hyperplane<T, 2> start_to_point =
          Eigen::Hyperplane<T, 2>::Through(line_seg.start, point);
  Vector2T point_on_line = line.projection(point);
  CHECK(ceres::IsFinite(point_on_line.x()));
  CHECK(ceres::IsFinite(point_on_line.y()));
  Vector2T line_length = (line_seg.endpoint - line_seg.start);
  CHECK(ceres::IsFinite(line_length.x()));
  CHECK(ceres::IsFinite(line_length.y()));
  // To not allow the differentiation to fail when the point gets close to the
  // beginning of the line as sqrt(x)'s derivative would go to inf as x
  // approaches 0.
  Vector2T distance_on_line = (point_on_line - line_seg.start);
  CHECK(ceres::IsFinite(distance_on_line.x()));
  CHECK(ceres::IsFinite(distance_on_line.y()));
  if (distance_on_line.x() >= T(0.0) && distance_on_line.x() <= line_length.x() &&
      distance_on_line.y() >= T(0.0) && distance_on_line.y() <= line_length.y()) {
    // Point is between start and end, should return perpendicular dist.
    Vector2T point_to_line = point_on_line - point;
    CHECK(ceres::IsFinite(point_to_line.x()));
    CHECK(ceres::IsFinite(point_to_line.y()));
    return pow(point_to_line.x(), 2) + pow(point_to_line.y(), 2);
  }
  // Point is closer to an endpoint.
  Vector2T point_to_start = line_seg.start - point;
  Vector2T point_to_endpoint = line_seg.endpoint - point;
  T dist_to_start = pow(point_to_start.x(), 2) + pow(point_to_endpoint.y(), 2);
  T dist_to_endpoint = pow(point_to_endpoint.x(), 2) +
     pow(point_to_endpoint.y(), 2);
  CHECK(ceres::IsFinite(dist_to_start));
  CHECK(ceres::IsFinite(dist_to_endpoint));
  return std::min<T>(dist_to_start, dist_to_endpoint);
}

struct LCPose {
    uint64_t node_idx;
    vector<Vector2f> points_on_feature;
    LCPose(uint64_t node_idx, vector<Vector2f> points_on_feature) :
      node_idx(node_idx), points_on_feature(points_on_feature) {}
};

struct LCConstraint {
    vector<LCPose> line_a_poses;
    vector<LCPose> line_b_poses;
    const LineSegment<float>& line_a;
    const LineSegment<float>& line_b;
    LCConstraint(const LineSegment<float>& line_a,
                 const LineSegment<float>& line_b) :
                 line_a(line_a),
                 line_b(line_b) {}
};

class Solver {
 public:
  struct PointCorrespondences {
    vector<Vector2f> source_points;
    vector<Vector2f> target_points;
    vector<Vector2f> source_normals;
    vector<Vector2f> target_normals;
    double *source_pose;
    double *target_pose;
    PointCorrespondences(double* source_pose, double* target_pose)
    : source_pose(source_pose), target_pose(target_pose) {}
  };
  Solver(double translation_weight,
         double rotation_weight,
         double lc_translation_weight,
         double lc_rotation_weight,
         double stopping_accuracy,
         SLAMProblem2D& problem,
         ros::NodeHandle& n);
  vector<SLAMNodeSolution2D> SolveSLAM();
  double GetPointCorrespondences(const SLAMProblem2D& problem,
                                 vector<SLAMNodeSolution2D>*
                                   solution_ptr,
                                 PointCorrespondences* point_correspondences,
                                 size_t source_node_index,
                                 size_t target_node_index);
  void AddOdomFactors(ceres::Problem* ceres_problem,
                      double trans_weight,
                      double rot_weight);
  void HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr);
  vector<SLAMNodeSolution2D> GetSolution() {
    return solution_;
  }
  LCConstraint GetRelevantPosesForHITL(const HitlSlamInputMsg& hitl_msg);
  void AddColinearConstraints(LCConstraint& constraint);
  void SolveForLC();
  void AddColinearResiduals(ceres::Problem* problem);
  double AddLidarResidualsForLC(ceres::Problem& problem);
private:
  double translation_weight_;
  double rotation_weight_;
  double lc_translation_weight_;
  double lc_rotation_weight_;
  double stopping_accuracy_;
  SLAMProblem2D problem_;
  vector<SLAMNodeSolution2D> solution_;
  ros::NodeHandle n_;
  vector<LCConstraint> loop_closure_constraints_;
};

#endif // SRC_SOLVER_H_
