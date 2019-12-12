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
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"

#include "./kdtree.h"
#include "./slam_types.h"
#include "lidar_slam/HitlSlamInputMsg.h"
#include "./pointcloud_helpers.h"
#include "./gui_helpers.h"

using std::vector;
using Eigen::Vector2f;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using slam_types::SLAMNode2D;
using lidar_slam::HitlSlamInputMsgConstPtr;
using lidar_slam::HitlSlamInputMsg;
using Eigen::Affine2f;
using Eigen::Vector3f;

template<typename T> Eigen::Transform<T, 2, Eigen::Affine>
PoseArrayToAffine(const T* rotation, const T* translation) {
  typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
  typedef Eigen::Rotation2D<T> Rotation2DT;
  typedef Eigen::Translation<T, 2> Translation2T;
  Affine2T affine = Translation2T(translation[0], translation[1]) *
                    Rotation2DT(rotation[0]).toRotationMatrix();
  return affine;
}

template <typename T>
struct LineSegment {
    const Eigen::Matrix<T, 2, 1> start;
    const Eigen::Matrix<T, 2, 1> endpoint;
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

// Returns if val is between a and b.
template <typename T>
T IsBetween(const T& val, const T& a, const T& b) {
  return (val >= a && val <= b) || (val >= b && val <= a);
}

template <typename T>
T DistanceToLineSegment(const Eigen::Matrix<T, 2, 1>& point,
                               const LineSegment<T>& line_seg) {
  typedef Eigen::Matrix<T, 2, 1> Vector2T;
  // Line segment is parametric, with a start point and endpoint.
  // Parameterized by t between 0 and 1.
  // We can get the point on the line by projecting the start -> point onto
  // this line.
  Eigen::Hyperplane<T, 2> line =
    Eigen::Hyperplane<T, 2>::Through(line_seg.start, line_seg.endpoint);
  Vector2T point_on_line = line.projection(point);
  if (IsBetween(point_on_line.x(), line_seg.start.x(), line_seg.endpoint.x()) &&
      IsBetween(point_on_line.y(), line_seg.start.y(), line_seg.endpoint.y())) {
    return line.absDistance(point);
  }

  T dist_to_start = (point - line_seg.start).norm();
  T dist_to_endpoint = (point - line_seg.endpoint).norm();
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
    const LineSegment<float> line_a;
    const LineSegment<float> line_b;
    LCConstraint(const LineSegment<float>& line_a,
                 const LineSegment<float>& line_b) :
                 line_a(line_a),
                 line_b(line_b) {}
};

struct PointCorrespondences {
    vector<Vector2f> source_points;
    vector<Vector2f> target_points;
    vector<Vector2f> source_normals;
    vector<Vector2f> target_normals;
    double *source_pose;
    double *target_pose;
    PointCorrespondences(double* source_pose, double* target_pose)
      : source_pose(source_pose), target_pose(target_pose) {}
    PointCorrespondences() : source_pose(nullptr), target_pose(nullptr) {}
};

class VisualizationCallback : public ceres::IterationCallback {
 public:
  VisualizationCallback(const SLAMProblem2D& problem,
                        const vector<SLAMNodeSolution2D>* solution,
                        ros::NodeHandle& n) :
          problem(problem),
          solution(solution) {
    pointcloud_helpers::InitPointcloud(&all_points_marker);
    pointcloud_helpers::InitPointcloud(&new_points_marker);
    all_points.clear();
    point_pub = n.advertise<sensor_msgs::PointCloud2>("/all_points", 10);
    new_point_pub = n.advertise<sensor_msgs::PointCloud2>("/new_points", 10);
    pose_pub = n.advertise<visualization_msgs::Marker>("/poses", 10);
    match_pub = n.advertise<visualization_msgs::Marker>("/matches", 10);
    normals_pub = n.advertise<visualization_msgs::Marker>("/normals", 10);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_STRIP,
                                  gui_helpers::Color4f::kGreen,
                                  0.002,
                                  0.0,
                                  0.0,
                                  &pose_array);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kBlue,
                                  0.003,
                                  0.0,
                                  0.0,
                                  &match_line_list);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kYellow,
                                  0.003,
                                  0.0,
                                  0.0,
                                  &normals_marker);
    pose_pub = n.advertise<PointCloud2>("/hitl_poses", 10);
    point_a_pub = n.advertise<PointCloud2>("/hitl_a_points",100);
    point_b_pub = n.advertise<PointCloud2>("/hitl_b_points",100);
    line_pub = n.advertise<visualization_msgs::Marker>("/line_a", 10);
    pointcloud_helpers::InitPointcloud(&pose_point_marker);
    pointcloud_helpers::InitPointcloud(&a_points_marker);
    pointcloud_helpers::InitPointcloud(&b_points_marker);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kMagenta,
                                  0.10,
                                  0.0,
                                  0.0,
                                  &line_marker);
  }

  void PubVisualization() {
    const vector<SLAMNodeSolution2D>& solution_c = *solution;
    vector<Vector2f> new_points;
    gui_helpers::ClearMarker(&pose_array);
    for (size_t i = 0; i < solution_c.size(); i++) {
      if (new_points.size() > 0) {
        all_points.insert(all_points.end(),
                          new_points.begin(),
                          new_points.end());
        new_points.clear();
      }
      auto pointcloud = problem.nodes[i].lidar_factor.pointcloud;
      Affine2f robot_to_world =
              PoseArrayToAffine(&(solution_c[i].pose[2]),
                                &(solution_c[i].pose[0])).cast<float>();
      Eigen::Vector3f pose(solution_c[i].pose[0], solution_c[i].pose[1], 0.0);
      gui_helpers::AddPoint(pose, gui_helpers::Color4f::kGreen, &pose_array);
      for (const Vector2f& point : pointcloud) {
        new_points.push_back(robot_to_world * point);
        // Visualize normal
        KDNodeValue<float, 2> source_point_in_tree;
        float dist =
                problem.nodes[i].lidar_factor
                        .pointcloud_tree->FindNearestPoint(point,
                                                           0.01,
                                                           &source_point_in_tree);
        if (dist != 0.01) {
          Eigen::Vector3f normal(source_point_in_tree.normal.x(),
                                 source_point_in_tree.normal.y(),
                                 0.0);
          normal = robot_to_world * normal;
          Vector2f source_point = robot_to_world * point;
          Eigen::Vector3f source_3f(source_point.x(), source_point.y(), 0.0);
          Eigen::Vector3f result = source_3f + (normal * 0.01);
          gui_helpers::AddLine(source_3f,
                               result,
                               gui_helpers::Color4f::kGreen,
                               &normals_marker);
        }
      }
    }
    if (solution_c.size() >= 2) {
      pointcloud_helpers::PublishPointcloud(all_points,
                                            all_points_marker,
                                            point_pub);
      pointcloud_helpers::PublishPointcloud(new_points,
                                            new_points_marker,
                                            new_point_pub);
      gui_helpers::ClearMarker(&match_line_list);
      AddMatchLines();
      pose_pub.publish(pose_array);
      match_pub.publish(match_line_list);
      normals_pub.publish(normals_marker);
    }
    all_points.clear();
    PubConstraintVisualization();
  }

  void PubConstraintVisualization() {
    const vector<SLAMNodeSolution2D>& solution_c = *solution;
    for (const LCConstraint& hitl_constraint : constraints) {
      vector<Vector2f> pose_points;
      vector<Vector2f> a_points;
      vector<Vector2f> b_points;
      for (const LCPose &pose : hitl_constraint.line_a_poses) {
        const double *pose_arr = solution_c[pose.node_idx].pose;
        Vector2f pose_pos(pose_arr[0], pose_arr[1]);
        pose_points.push_back(pose_pos);
        Affine2f point_to_world = PoseArrayToAffine(&pose_arr[2],
                                                    &pose_arr[0]).cast<float>();
        for (const Vector2f &point : pose.points_on_feature) {
          Vector2f point_transformed = point_to_world * point;
          a_points.push_back(point_transformed);
        }
      }
      for (const LCPose &pose : hitl_constraint.line_b_poses) {
        const double *pose_arr = solution_c[pose.node_idx].pose;
        Vector2f pose_pos(pose_arr[0], pose_arr[1]);
        pose_points.push_back(pose_pos);
        Affine2f point_to_world = PoseArrayToAffine(&pose_arr[2],
                                                    &pose_arr[0]).cast<float>();
        for (const Vector2f &point : pose.points_on_feature) {
          Vector2f point_transformed = point_to_world * point;
          b_points.push_back(point_transformed);
        }
      }
      gui_helpers::AddLine(Vector3f(hitl_constraint.line_a.start.x(),
                                    hitl_constraint.line_a.start.y(),
                                    0.0),
                           Vector3f(hitl_constraint.line_a.endpoint.x(),
                                    hitl_constraint.line_a.endpoint.y(),
                                    0.0),
                           gui_helpers::Color4f::kMagenta,
                           &line_marker);
      for (int i = 0; i < 5; i++) {
        pointcloud_helpers::PublishPointcloud(pose_points, pose_point_marker,
                                              constraint_pose_pub);
        pointcloud_helpers::PublishPointcloud(a_points, a_points_marker,
                                              point_a_pub);
        pointcloud_helpers::PublishPointcloud(b_points, b_points_marker,
                                              point_b_pub);
        line_pub.publish(line_marker);
        sleep(1);
      }
    }
  }

  void ClearNormals() {
    gui_helpers::ClearMarker(&normals_marker);
  }

  void AddMatchLines() {
    CHECK_EQ(last_correspondence.source_points.size(),
             last_correspondence.target_points.size());
    for (uint64_t index = 0;
         index < last_correspondence.source_points.size();
         index++) {
      Vector3f source_3d(last_correspondence.source_points[index].x(),
                         last_correspondence.source_points[index].y(),
                         0.0);
      Vector3f target_3d(last_correspondence.target_points[index].x(),
                         last_correspondence.target_points[index].y(),
                         0.0);
      gui_helpers::AddLine(source_3d,
                           target_3d,
                           gui_helpers::Color4f::kBlue,
                           &match_line_list);
    }
  }

  void UpdateLastCorrespondence(PointCorrespondences& point_correspondence) {
    last_correspondence = point_correspondence;
  }

  void AddConstraint(const LCConstraint& constraint) {
    constraints.push_back(constraint);
  }

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
  override {
    PubVisualization();
    return ceres::SOLVER_CONTINUE;
  }

 private:
  sensor_msgs::PointCloud2 all_points_marker;
  sensor_msgs::PointCloud2 new_points_marker;
  std::vector<Vector2f> all_points;
  const SLAMProblem2D& problem;
  const vector<SLAMNodeSolution2D>* solution;
  ros::Publisher point_pub;
  ros::Publisher pose_pub;
  ros::Publisher match_pub;
  ros::Publisher new_point_pub;
  ros::Publisher normals_pub;
  ros::Publisher constraint_pose_pub;
  ros::Publisher point_a_pub;
  ros::Publisher point_b_pub;
  ros::Publisher line_pub;
  visualization_msgs::Marker pose_array;
  visualization_msgs::Marker match_line_list;
  visualization_msgs::Marker normals_marker;
  PointCloud2 pose_point_marker;
  PointCloud2 a_points_marker;
  PointCloud2 b_points_marker;
  visualization_msgs::Marker line_marker;
  PointCorrespondences last_correspondence;
  vector<LCConstraint> constraints;
};

class Solver {
 public:
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
  void AddColinearConstraints(const LCConstraint& constraint);
  void SolveForLC();
  void AddCollinearResiduals(ceres::Problem* problem);
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
  std::unique_ptr<VisualizationCallback> vis_callback_ = nullptr;
};

#endif // SRC_SOLVER_H_
