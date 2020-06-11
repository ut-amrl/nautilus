//
// Created by jack on 9/25/19.
//

#ifndef SRC_SOLVER_H_
#define SRC_SOLVER_H_

#include <vector>

#include <ros/node_handle.h>
#include <boost/math/distributions/chi_squared.hpp>
#include "CorrelativeScanMatcher.h"
#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "glog/logging.h"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "geometry_msgs/PoseArray.h"

#include "./gui_helpers.h"
#include "./kdtree.h"
#include "./pointcloud_helpers.h"
#include "./slam_types.h"
#include "config_reader/config_reader.h"
#include "nautilus/HitlSlamInputMsg.h"
#include "nautilus/WriteMsg.h"

using boost::math::chi_squared;
using boost::math::complement;
using boost::math::quantile;
using Eigen::Affine2f;
using Eigen::Vector2f;
using Eigen::Vector3f;
using nautilus::HitlSlamInputMsg;
using nautilus::HitlSlamInputMsgConstPtr;
using nautilus::WriteMsgConstPtr;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNode2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using slam_types::LidarFactor;
using std::vector;

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
  vector<Vector2f> points_on_feature;
  LCPose(uint64_t node_idx, vector<Vector2f> points_on_feature)
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
  const SLAMNode2D* node_a;
  const SLAMNode2D* node_b;
  double source_pose[3];
  double target_pose[3];
  float match_ratio;
  Vector3f relative_transformation;
};

struct PointCorrespondences {
  vector<Vector2f> source_points;
  vector<Vector2f> target_points;
  vector<Vector2f> source_normals;
  vector<Vector2f> target_normals;
  double* source_pose;
  double* target_pose;
  uint64_t source_index;
  uint64_t target_index;
  PointCorrespondences(double* source_pose, double* target_pose,
                       uint64_t source_index, uint64_t target_index)
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
  vector<ResidualDesc> res_descriptors;
};

struct SolverConfig {
  CONFIG_DOUBLE(translation_weight, "translation_weight");
  CONFIG_DOUBLE(rotation_weight, "rotation_weight");
  CONFIG_DOUBLE(lc_translation_weight, "lc_translation_weight");
  CONFIG_DOUBLE(lc_rotation_weight, "lc_rotation_weight");
  CONFIG_DOUBLE(lc_base_max_range, "lc_base_max_range");
  CONFIG_DOUBLE(lc_max_range_scaling, "lc_max_range_scaling");
  CONFIG_STRING(lc_debug_output_dir, "lc_debug_output_dir");
  CONFIG_STRING(pose_output_file, "pose_output_file");
  CONFIG_STRING(map_output_file, "map_output_file");
  CONFIG_DOUBLE(stopping_accuracy, "stopping_accuracy");
  CONFIG_DOUBLE(max_lidar_range, "max_lidar_range");
  CONFIG_DOUBLE(lc_match_threshold, "lc_match_threshold");
  CONFIG_INT(lidar_constraint_amount, "lidar_constraint_amount");
  CONFIG_DOUBLE(outlier_threshold, "outlier_threshold");
  CONFIG_DOUBLE(hitl_line_width, "hitl_line_width");
  CONFIG_INT(hitl_pose_point_threshold, "hitl_pose_point_threshold");

  // Auto LC configs
  CONFIG_BOOL(keyframe_local_uncertainty_filtering, "keyframe_local_uncertainty_filtering");
  CONFIG_BOOL(keyframe_chi_squared_test, "keyframe_chi_squared_test");
  CONFIG_DOUBLE(keyframe_min_odom_distance, "keyframe_min_odom_distance");
  CONFIG_DOUBLE(local_uncertainty_condition_threshold,
                "local_uncertainty_condition_threshold");
  CONFIG_DOUBLE(local_uncertainty_scale_threshold,
                "local_uncertainty_scale_threshold");
  CONFIG_INT(local_uncertainty_prev_scans, "local_uncertainty_prev_scans");
  CONFIG_INT(lc_min_keyframes, "lc_min_keyframes");
  CONFIG_DOUBLE(csm_score_threshold, "csm_score_threshold");
  CONFIG_DOUBLE(translation_std_dev, "translation_standard_deviation");
  CONFIG_DOUBLE(rotation_std_dev, "rotation_standard_deviation");

  SolverConfig() {
    std::cout << "Solver Waiting..." << std::endl;
    config_reader::WaitForInit();
    std::cout << "--- Done Waiting ---" << std::endl;
  }
};

/*----------------------------------------------------------------------------*
 *                            HELPER FUNCTIONS                                |
 *----------------------------------------------------------------------------*/

template <typename T>
Eigen::Transform<T, 2, Eigen::Affine> PoseArrayToAffine(const T* rotation,
                                                        const T* translation) {
  typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
  typedef Eigen::Rotation2D<T> Rotation2DT;
  typedef Eigen::Translation<T, 2> Translation2T;
  Affine2T affine = Translation2T(translation[0], translation[1]) *
                    Rotation2DT(rotation[0]).toRotationMatrix();
  return affine;
}

template <typename T>
Eigen::Transform<T, 2, Eigen::Affine> PoseArrayToAffine(const T* pose_array) {
  return PoseArrayToAffine(&pose_array[2], &pose_array[0]);
}

inline double DistBetween(double pose_a[3], double pose_b[3]) {
  Vector2f pose_a_pos(pose_a[0], pose_a[1]);
  Vector2f pose_b_pos(pose_b[0], pose_b[1]);
  return (pose_a_pos - pose_b_pos).norm();
}

inline vector<Vector2f> TransformPointcloud(double* pose,
                                            const vector<Vector2f> pointcloud) {
  vector<Vector2f> pcloud;
  Eigen::Affine2f trans = PoseArrayToAffine(&pose[2], &pose[0]).cast<float>();
  for (const Vector2f& p : pointcloud) {
    pcloud.push_back(trans * p);
  }
  return pcloud;
}

// Reference from:
// https://github.com/SoylentGraham/libmv/blob/master/src/libmv/simple_pipeline/bundle.cc
inline Eigen::MatrixXd CRSToEigen(const ceres::CRSMatrix& crs_matrix) {
  Eigen::MatrixXd matrix(crs_matrix.num_rows, crs_matrix.num_cols);
  matrix.setZero();
  // Row contains starting position of this row in the cols.
  for (int row = 0; row < crs_matrix.num_rows; row++) {
    int row_start = crs_matrix.rows[row];
    int row_end = crs_matrix.rows[row + 1];
    // Cols contains the non-zero elements column numbers.
    for (int col = row_start; col < row_end; col++) {
      int col_num = crs_matrix.cols[col];
      // Value is contained in the same index of the values array.
      double value = crs_matrix.values[col];
      matrix(row, col_num) = value;
    }
  }
  return matrix;
}
struct LearnedKeyframe {
    const size_t node_idx;
    LearnedKeyframe(const size_t node_idx) :
                        node_idx(node_idx) {}
};

// Returns if val is between a and b.
template <typename T>
bool IsBetween(const T& val, const T& a, const T& b) {
  return (val >= a && val <= b) || (val >= b && val <= a);
}

template <typename T>
T DistanceToLineSegment(const Eigen::Matrix<T, 2, 1>& point,
                        const LineSegment<T>& line_seg) {
  typedef Eigen::Matrix<T, 2, 1> Vector2T;
  // Line segment is parametric, with a start point and end.
  // Parameterized by t between 0 and 1.
  // We can get the point on the line by projecting the start -> point onto
  // this line.
  Eigen::Hyperplane<T, 2> line =
      Eigen::Hyperplane<T, 2>::Through(line_seg.start, line_seg.end);
  Vector2T point_on_line = line.projection(point);
  if (IsBetween(point_on_line.x(), line_seg.start.x(), line_seg.end.x()) &&
      IsBetween(point_on_line.y(), line_seg.start.y(), line_seg.end.y())) {
    return line.absDistance(point);
  }

  T dist_to_start = (point - line_seg.start).norm();
  T dist_to_endpoint = (point - line_seg.end).norm();
  return std::min<T>(dist_to_start, dist_to_endpoint);
}

/*----------------------------------------------------------------------------*
 *                           DEBUGGING OUTPUT                                 |
 *----------------------------------------------------------------------------*/

class VisualizationCallback : public ceres::IterationCallback {
 public:
  VisualizationCallback(vector<LearnedKeyframe>& keyframes, ros::NodeHandle& n)
      : keyframes(keyframes) {
    pointcloud_helpers::InitPointcloud(&all_points_marker);
    pointcloud_helpers::InitPointcloud(&new_points_marker);
    pointcloud_helpers::InitPointcloud(&keyframe_marker);
    all_points.clear();
    point_pub = n.advertise<sensor_msgs::PointCloud2>("/all_points", 10);
    new_point_pub = n.advertise<sensor_msgs::PointCloud2>("/new_points", 10);
    pose_pub = n.advertise<geometry_msgs::PoseArray>("/poses", 10);
    keyframe_poses_pub =
        n.advertise<visualization_msgs::Marker>("/keyframe_poses", 10);
    match_pub = n.advertise<visualization_msgs::Marker>("/matches", 10);
    normals_pub = n.advertise<visualization_msgs::Marker>("/normals", 10);
    keyframe_pub = n.advertise<sensor_msgs::PointCloud2>("/keyframes", 10);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kBlue, 0.003, 0.0, 0.0,
                                  &match_line_list);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kYellow, 0.003, 0.0,
                                  0.0, &normals_marker);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kRed, 0.003, 0.0, 0.0,
                                  &key_pose_array);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kCyan, 0.01, 0.0, 0.0,
                                  &auto_lc_pose_array);
    pose_array.header.frame_id = "map";
    constraint_a_pose_pub = n.advertise<PointCloud2>("/hitl_poses_line_a", 10);
    constraint_b_pose_pub = n.advertise<PointCloud2>("/hitl_poses_line_b", 10);
    hitl_pointclouds = n.advertise<PointCloud2>("/hitl_pointclouds", 10);
    point_a_pub = n.advertise<PointCloud2>("/hitl_a_points", 100);
    point_b_pub = n.advertise<PointCloud2>("/hitl_b_points", 100);
    auto_lc_poses_pub = n.advertise<visualization_msgs::Marker>("/auto_lc_poses", 10);
    line_pub = n.advertise<visualization_msgs::Marker>("/line_a", 10);
    pointcloud_helpers::InitPointcloud(&pose_a_point_marker);
    pointcloud_helpers::InitPointcloud(&pose_b_point_marker);
    pointcloud_helpers::InitPointcloud(&a_points_marker);
    pointcloud_helpers::InitPointcloud(&b_points_marker);
    pointcloud_helpers::InitPointcloud(&hitl_points_marker);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kMagenta, 0.10, 0.0,
                                  0.0, &line_marker);
  }

  void PubVisualization() {
    const vector<SLAMNodeSolution2D>& solution_c = *solution;
    vector<Vector2f> new_points;
    gui_helpers::ClearMarker(&key_pose_array);
    pose_array.poses.clear();
    for (size_t i = 0; i < solution_c.size(); i++) {
      if (new_points.size() > 0) {
        all_points.insert(all_points.end(), new_points.begin(),
                          new_points.end());
        new_points.clear();
      }
      auto pointcloud = problem.nodes[i].lidar_factor.pointcloud;
      Affine2f robot_to_world =
          PoseArrayToAffine(&(solution_c[i].pose[2]), &(solution_c[i].pose[0]))
              .cast<float>();
      Eigen::Vector3f pose(solution_c[i].pose[0], solution_c[i].pose[1], 0.0);
      // gui_helpers::AddPoint(pose, gui_helpers::Color4f::kGreen, &pose_array);
      geometry_msgs::Pose p;
      p.position.x = solution_c[i].pose[0];
      p.position.y = solution_c[i].pose[1];
      p.orientation.z = sin(solution_c[i].pose[2]/2);
      p.orientation.w = cos(solution_c[i].pose[2]/2);
      pose_array.poses.push_back(p);
      if (solution_c[i].is_keyframe) {
        gui_helpers::AddPoint(pose, gui_helpers::Color4f::kRed,
                              &key_pose_array);
      }
      gui_helpers::ClearMarker(&normals_marker);
      for (const Vector2f& point : pointcloud) {
        new_points.push_back(robot_to_world * point);
        // Visualize normal
        KDNodeValue<float, 2> source_point_in_tree;
        float dist =
            problem.nodes[i].lidar_factor.pointcloud_tree->FindNearestPoint(
                point, 0.01, &source_point_in_tree);
        if (dist != 0.01) {
          Eigen::Vector3f normal(source_point_in_tree.normal.x(),
                                 source_point_in_tree.normal.y(), 0.0);
          normal = robot_to_world * normal;
          Vector2f source_point = robot_to_world * point;
          Eigen::Vector3f source_3f(source_point.x(), source_point.y(), 0.0);
          Eigen::Vector3f result = source_3f + (normal * 0.01);
          gui_helpers::AddLine(source_3f, result, gui_helpers::Color4f::kGreen,
                               &normals_marker);
        }
      }
    }
    if (solution_c.size() >= 2) {
      pointcloud_helpers::PublishPointcloud(all_points, all_points_marker,
                                            point_pub);
      pointcloud_helpers::PublishPointcloud(new_points, new_points_marker,
                                            new_point_pub);
      gui_helpers::ClearMarker(&match_line_list);
      for (const PointCorrespondences& corr : last_correspondences) {
        AddMatchLines(corr);
      }
      pose_pub.publish(pose_array);
      match_pub.publish(match_line_list);
      normals_pub.publish(normals_marker);
      keyframe_poses_pub.publish(key_pose_array);
    }
    all_points.clear();
    PubConstraintVisualization();
    PubAutoConstraints();
  }

  void PubConstraintVisualization() {
    const vector<SLAMNodeSolution2D>& solution_c = *solution;
    vector<Vector2f> line_a_poses;
    for (const HitlLCConstraint& hitl_constraint : hitl_constraints) {
      vector<Vector2f> a_points;
      vector<Vector2f> b_points;
      for (const LCPose& pose : hitl_constraint.line_a_poses) {
        const double* pose_arr = solution_c[pose.node_idx].pose;
        Vector2f pose_pos(pose_arr[0], pose_arr[1]);
        line_a_poses.push_back(pose_pos);
        Affine2f point_to_world =
            PoseArrayToAffine(&pose_arr[2], &pose_arr[0]).cast<float>();
        for (const Vector2f& point : pose.points_on_feature) {
          Vector2f point_transformed = point_to_world * point;
          a_points.push_back(point_transformed);
        }
      }
      vector<Vector2f> line_b_poses;
      for (const LCPose& pose : hitl_constraint.line_b_poses) {
        const double* pose_arr = solution_c[pose.node_idx].pose;
        Vector2f pose_pos(pose_arr[0], pose_arr[1]);
        line_b_poses.push_back(pose_pos);
        Affine2f point_to_world =
            PoseArrayToAffine(&pose_arr[2], &pose_arr[0]).cast<float>();
        for (const Vector2f& point : pose.points_on_feature) {
          Vector2f point_transformed = point_to_world * point;
          b_points.push_back(point_transformed);
        }
      }
      gui_helpers::AddLine(Vector3f(hitl_constraint.line_a.start.x(),
                                    hitl_constraint.line_a.start.y(), 0.0),
                           Vector3f(hitl_constraint.line_a.end.x(),
                                    hitl_constraint.line_a.end.y(), 0.0),
                           gui_helpers::Color4f::kMagenta, &line_marker);
      for (int i = 0; i < 5; i++) {
        pointcloud_helpers::PublishPointcloud(line_a_poses, pose_a_point_marker,
                                              constraint_a_pose_pub);
        pointcloud_helpers::PublishPointcloud(line_b_poses, pose_b_point_marker,
                                              constraint_b_pose_pub);
        pointcloud_helpers::PublishPointcloud(a_points, a_points_marker,
                                              point_a_pub);
        pointcloud_helpers::PublishPointcloud(b_points, b_points_marker,
                                              point_b_pub);
        line_pub.publish(line_marker);
      }
    }
    PubConstraintPointclouds();
    PubKeyframes();
  }

  void PubKeyframes() {
    vector<Vector2f> poses;
    for (LearnedKeyframe frame : keyframes) {
      // std::cout << "Frame #: " << frame.node_idx << std::endl;
      const double* pose_arr = (*solution)[frame.node_idx].pose;
      Vector2f pose_point(pose_arr[0], pose_arr[1]);
      poses.push_back(pose_point);
    }
    pointcloud_helpers::PublishPointcloud(poses, keyframe_marker, keyframe_pub);
  }

  void AddPosePointcloud(vector<Vector2f>& pointcloud, const LCPose& pose) {
    size_t node_idx = pose.node_idx;
    vector<SLAMNodeSolution2D> solution_c = *solution;
    vector<Vector2f> p_cloud = problem.nodes[node_idx].lidar_factor.pointcloud;
    double* pose_arr = solution_c[node_idx].pose;
    Affine2f robot_to_world =
        PoseArrayToAffine(&pose_arr[2], &pose_arr[0]).cast<float>();
    for (const Vector2f& p : p_cloud) {
      pointcloud.push_back(robot_to_world * p);
    }
  }

  void PubConstraintPointclouds() {
    vector<Vector2f> pointclouds;
    for (const HitlLCConstraint& hitl_constraint : hitl_constraints) {
      for (const LCPose& pose : hitl_constraint.line_a_poses) {
        AddPosePointcloud(pointclouds, pose);
      }
      for (const LCPose& pose : hitl_constraint.line_b_poses) {
        AddPosePointcloud(pointclouds, pose);
      }
    }
    pointcloud_helpers::PublishPointcloud(pointclouds, hitl_points_marker,
                                          hitl_pointclouds);
  }

  void PubAutoConstraints() {
    vector<Vector3f> poses;
    gui_helpers::ClearMarker(&auto_lc_pose_array);
    for (const AutoLCConstraint& auto_constraint : auto_constraints) {
      // std::cout << "Frame #: " << frame.node_idx << std::endl;
      const double* pose_arr_a = (*solution)[auto_constraint.node_a->node_idx].pose;
      Vector3f pose_a(pose_arr_a[0], pose_arr_a[1], 0.0);
      gui_helpers::AddPoint(pose_a, gui_helpers::Color4f::kCyan, &auto_lc_pose_array);
      const double* pose_arr_b = (*solution)[auto_constraint.node_b->node_idx].pose;
      Vector3f pose_b(pose_arr_b[0], pose_arr_b[1], 0.0);
      gui_helpers::AddPoint(pose_b, gui_helpers::Color4f::kCyan, &auto_lc_pose_array);
    }
    if (auto_constraints.size() > 0) {
      auto_lc_poses_pub.publish(auto_lc_pose_array);
    }
  }

  void ClearNormals() { gui_helpers::ClearMarker(&normals_marker); }

  void AddMatchLines(const PointCorrespondences& correspondence) {
    CHECK_EQ(correspondence.source_points.size(),
             correspondence.target_points.size());
    for (uint64_t index = 0; index < correspondence.source_points.size();
         index++) {
      Vector2f source_point = correspondence.source_points[index];
      Vector2f target_point = correspondence.target_points[index];
      Affine2f source_to_world =
          PoseArrayToAffine(correspondence.source_pose).cast<float>();
      Affine2f target_to_world =
          PoseArrayToAffine(correspondence.target_pose).cast<float>();
      source_point = source_to_world * source_point;
      target_point = target_to_world * target_point;
      Vector3f source_3d(source_point.x(), source_point.y(), 0.0);
      Vector3f target_3d(target_point.x(), target_point.y(), 0.0);
      gui_helpers::AddLine(source_3d, target_3d, gui_helpers::Color4f::kBlue,
                           &match_line_list);
    }
  }

  void UpdateLastCorrespondence(PointCorrespondences& point_correspondence) {
    // Comparing literal addresses because the same pose
    // points to the same place.
    if (!last_correspondences.empty() &&
        point_correspondence.source_index !=
            last_correspondences[0].source_index) {
      last_correspondences.clear();
    }
    last_correspondences.push_back(point_correspondence);
  }

  void AddConstraint(const HitlLCConstraint& constraint) {
    hitl_constraints.push_back(constraint);
  }

  void AddAutoLCConstraint(const AutoLCConstraint& constraint) {
    auto_constraints.push_back(constraint);
  }

  void AddKeyframe(const LearnedKeyframe& keyframe) {
    keyframes.push_back(keyframe);
  }

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    PubVisualization();
    return ceres::SOLVER_CONTINUE;
  }

  void UpdateProblemAndSolution(SLAMNode2D& new_node,
                                vector<SLAMNodeSolution2D>* new_solution,
                                OdometryFactor2D& new_odom_factor) {
    CHECK_EQ(new_node.node_idx,
             (*new_solution)[new_solution->size() - 1].node_idx);
    CHECK_EQ(new_node.node_idx, new_odom_factor.pose_j);
    problem.nodes.push_back(new_node);
    problem.odometry_factors.push_back(new_odom_factor);
    solution = new_solution;
    CHECK_EQ(solution->size(), problem.nodes.size());
  }

  void UpdateProblemAndSolution(SLAMNode2D& new_node,
                                vector<SLAMNodeSolution2D>* new_solution) {
    CHECK_EQ(new_node.node_idx,
             (*new_solution)[new_solution->size() - 1].node_idx);
    problem.nodes.push_back(new_node);
    solution = new_solution;
    CHECK_EQ(solution->size(), problem.nodes.size());
  }

 private:
  sensor_msgs::PointCloud2 all_points_marker;
  sensor_msgs::PointCloud2 new_points_marker;
  std::vector<Vector2f> all_points;
  SLAMProblem2D problem;
  vector<SLAMNodeSolution2D>* solution;
  vector<LearnedKeyframe>& keyframes;
  ros::Publisher point_pub;
  ros::Publisher pose_pub;
  ros::Publisher match_pub;
  ros::Publisher new_point_pub;
  ros::Publisher normals_pub;
  ros::Publisher constraint_a_pose_pub;
  ros::Publisher constraint_b_pose_pub;
  ros::Publisher keyframe_poses_pub;
  ros::Publisher auto_lc_poses_pub;
  ros::Publisher point_a_pub;
  ros::Publisher point_b_pub;
  ros::Publisher line_pub;
  ros::Publisher hitl_pointclouds;
  ros::Publisher keyframe_pub;
  geometry_msgs::PoseArray pose_array;
  visualization_msgs::Marker key_pose_array;
  visualization_msgs::Marker auto_lc_pose_array;
  visualization_msgs::Marker match_line_list;
  visualization_msgs::Marker normals_marker;
  PointCloud2 pose_a_point_marker;
  PointCloud2 pose_b_point_marker;
  PointCloud2 a_points_marker;
  PointCloud2 b_points_marker;
  PointCloud2 hitl_points_marker;
  PointCloud2 keyframe_marker;
  visualization_msgs::Marker line_marker;
  // All the correspondences were the source is the same
  // (will be the last pointcloud aligned and all of its targets).
  vector<PointCorrespondences> last_correspondences;
  vector<HitlLCConstraint> hitl_constraints;
  vector<AutoLCConstraint> auto_constraints;
};

/*----------------------------------------------------------------------------*
 *                                SOLVER                                      |
 *----------------------------------------------------------------------------*/

class Solver {
 public:
  Solver(ros::NodeHandle& n);
  vector<SLAMNodeSolution2D> SolveSLAM();
  vector<SLAMNodeSolution2D> SolvePoseSLAM();
  double GetPointCorrespondences(const LidarFactor& source_lidar, const LidarFactor& target_lidar,
    double* source_pose, double* target_pose, PointCorrespondences* point_correspondences);
  void AddOdomFactors(ceres::Problem* ceres_problem,
                      vector<OdometryFactor2D> factors, double trans_weight,
                      double rot_weight);
  void HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr);
  void WriteCallback(const WriteMsgConstPtr& msg);
  void Vectorize(const WriteMsgConstPtr& msg);
  vector<SLAMNodeSolution2D> GetSolution() { return solution_; }
  HitlLCConstraint GetRelevantPosesForHITL(const HitlSlamInputMsg& hitl_msg);
  void SolveForLC();
  double AddResidualsForAutoLC(ceres::Problem* problem, bool include_lidar);
  void AddPointCloudResiduals(ceres::Problem* problem);
  vector<OdometryFactor2D> GetSolvedOdomFactors();
  void AddSLAMNodeOdom(SLAMNode2D& node, OdometryFactor2D& odom_factor_to_node);
  void AddSlamNode(SLAMNode2D& node);
  void CheckForLearnedLC(SLAMNode2D& node);

 private:
  double CostFromResidualDescriptor(const ResidualDesc& res_desc);
  double GetChiSquareCost(uint64_t node_a, uint64_t node_b);
  OdometryFactor2D GetDifferenceOdom(const uint64_t node_a,
                                     const uint64_t node_b);
  OdometryFactor2D GetDifferenceOdom(const uint64_t node_a,
                                     const uint64_t node_b,
                                     Vector3f trans);
  vector<ResidualDesc> AddLCResiduals(const uint64_t node_a,
                                      const uint64_t node_b);
  void AddHITLResiduals(ceres::Problem* problem);
  void RemoveResiduals(vector<ResidualDesc> descs);
  void AddKeyframe(SLAMNode2D& node);
  float GetMatchScores(SLAMNode2D& node, SLAMNode2D& keyframe);
  bool AddKeyframeResiduals(LearnedKeyframe& key_frame_a,
                            LearnedKeyframe& key_frame_b);
  AutoLCConstraint computeAutoLCConstraint(const uint64_t node_a, const uint64_t node_b);
  bool AddAutoLCConstraint(const AutoLCConstraint& constraint);
  void LCKeyframes(LearnedKeyframe& key_frame_a, LearnedKeyframe& key_frame_b);
  OdometryFactor2D GetTotalOdomChange(const uint64_t node_a,
                                      const uint64_t node_b);
  bool SimilarScans(const uint64_t node_a, const uint64_t node_b,
                    const double certainty);
  std::pair<double, double> GetLocalUncertainty(const uint64_t node_idx);
  vector<size_t> GetMatchingKeyframeIndices(size_t keyframe_index);
  SLAMProblem2D problem_;
  vector<OdometryFactor2D> initial_odometry_factors;
  vector<SLAMNodeSolution2D> solution_;
  ros::NodeHandle n_;
  vector<AutoLCConstraint> auto_lc_constraints_;
  vector<HitlLCConstraint> hitl_constraints_;
  std::unique_ptr<VisualizationCallback> vis_callback_ = nullptr;
  vector<LearnedKeyframe> keyframes;
  ros::ServiceClient matcher_client;
  CorrelativeScanMatcher scan_matcher;
  CeresInformation ceres_information;
  SolverConfig config_;
};

#endif  // SRC_SOLVER_H_
