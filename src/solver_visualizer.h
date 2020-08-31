#pragma once
#include "ceres/ceres.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"

#include "solver_datastructures.h"

namespace nautilus {

/*----------------------------------------------------------------------------*
 *                           DEBUGGING OUTPUT                                 |
 *----------------------------------------------------------------------------*/

class VisualizationCallback : public ceres::IterationCallback {
 public:
  VisualizationCallback(std::vector<ds::LearnedKeyframe>& keyframes,
                        ros::NodeHandle& n)
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
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kRed,
                                  0.003,
                                  0.0,
                                  0.0,
                                  &key_pose_array);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kCyan,
                                  0.01,
                                  0.0,
                                  0.0,
                                  &auto_lc_pose_array);
    pose_array.header.frame_id = "map";
    constraint_a_pose_pub = n.advertise<PointCloud2>("/hitl_poses_line_a", 10);
    constraint_b_pose_pub = n.advertise<PointCloud2>("/hitl_poses_line_b", 10);
    hitl_pointclouds = n.advertise<PointCloud2>("/hitl_pointclouds", 10);
    point_a_pub = n.advertise<PointCloud2>("/hitl_a_points", 100);
    point_b_pub = n.advertise<PointCloud2>("/hitl_b_points", 100);
    line_pub = n.advertise<visualization_msgs::Marker>("/line_a", 10);
    pointcloud_helpers::InitPointcloud(&pose_a_point_marker);
    pointcloud_helpers::InitPointcloud(&pose_b_point_marker);
    pointcloud_helpers::InitPointcloud(&a_points_marker);
    pointcloud_helpers::InitPointcloud(&b_points_marker);
    pointcloud_helpers::InitPointcloud(&hitl_points_marker);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kMagenta,
                                  0.10,
                                  0.0,
                                  0.0,
                                  &line_marker);
  }

  void PubVisualization() {
    const std::vector<slam_types::SLAMNodeSolution2D>& solution_c = *solution;
    std::vector<Eigen::Vector2f> new_points;
    gui_helpers::ClearMarker(&key_pose_array);
    pose_array.poses.clear();
    CHECK_EQ(problem.nodes.size(), solution_c.size());
    for (size_t i = 0; i < solution_c.size(); ++i) {
      const auto& node = solution_c[i];
      const auto& problem_node = problem.nodes[i];
      if (!new_points.empty()) {
        all_points.insert(
            all_points.end(), new_points.begin(), new_points.end());
        new_points.clear();
      }
      const auto& pointcloud = problem_node.lidar_factor.pointcloud;
      Eigen::Affine2f robot_to_world =
          PoseArrayToAffine(node.pose).cast<float>();
      Eigen::Vector3f pose(node.pose[0], node.pose[1], 0.0);
      // gui_helpers::AddPoint(pose, gui_helpers::Color4f::kGreen, &pose_array);
      geometry_msgs::Pose p;
      p.position.x = node.pose[0];
      p.position.y = node.pose[1];
      p.orientation.z = sin(node.pose[2] / 2);
      p.orientation.w = cos(node.pose[2] / 2);
      pose_array.poses.push_back(p);
      if (node.is_keyframe) {
        gui_helpers::AddPoint(
            pose, gui_helpers::Color4f::kRed, &key_pose_array);
      }
      gui_helpers::ClearMarker(&normals_marker);
      for (const Eigen::Vector2f& point : pointcloud) {
        new_points.push_back(robot_to_world * point);
        // Visualize normal
        KDNodeValue<float, 2> source_point_in_tree;
        static constexpr float kMinDist = 0.01;
        float dist =
            problem_node.lidar_factor.pointcloud_tree->FindNearestPoint(
                point, kMinDist, &source_point_in_tree);
        if (dist != kMinDist) {
          Eigen::Vector3f normal(source_point_in_tree.normal.x(),
                                 source_point_in_tree.normal.y(),
                                 0.0);
          normal = robot_to_world * normal;
          Eigen::Vector2f source_point = robot_to_world * point;
          Eigen::Vector3f source_3f(source_point.x(), source_point.y(), 0.0);
          Eigen::Vector3f result = source_3f + (normal * 0.01);
          gui_helpers::AddLine(
              source_3f, result, gui_helpers::Color4f::kGreen, &normals_marker);
        }
      }
    }
    if (solution_c.size() >= 2) {
      pointcloud_helpers::PublishPointcloud(
          all_points, all_points_marker, point_pub);
      pointcloud_helpers::PublishPointcloud(
          new_points, new_points_marker, new_point_pub);
      gui_helpers::ClearMarker(&match_line_list);
      pose_pub.publish(pose_array);
      match_pub.publish(match_line_list);
      normals_pub.publish(normals_marker);
      keyframe_poses_pub.publish(key_pose_array);
    }
    all_points.clear();
    PubConstraintVisualization();
  }

  void PubConstraintVisualization() {
    const std::vector<slam_types::SLAMNodeSolution2D>& solution_c = *solution;
    std::vector<Eigen::Vector2f> line_a_poses;
    for (const ds::HitlLCConstraint& hitl_constraint : hitl_constraints) {
      std::vector<Eigen::Vector2f> a_points;
      std::vector<Eigen::Vector2f> b_points;
      for (const ds::LCPose& pose : hitl_constraint.line_a_poses) {
        const double* pose_arr = solution_c[pose.node_idx].pose;
        Eigen::Vector2f pose_pos(pose_arr[0], pose_arr[1]);
        line_a_poses.push_back(pose_pos);
        Eigen::Affine2f point_to_world =
            PoseArrayToAffine(&pose_arr[2], &pose_arr[0]).cast<float>();
        for (const Eigen::Vector2f& point : pose.points_on_feature) {
          Eigen::Vector2f point_transformed = point_to_world * point;
          a_points.push_back(point_transformed);
        }
      }
      std::vector<Eigen::Vector2f> line_b_poses;
      for (const ds::LCPose& pose : hitl_constraint.line_b_poses) {
        const double* pose_arr = solution_c[pose.node_idx].pose;
        Eigen::Vector2f pose_pos(pose_arr[0], pose_arr[1]);
        line_b_poses.push_back(pose_pos);
        Eigen::Affine2f point_to_world =
            PoseArrayToAffine(&pose_arr[2], &pose_arr[0]).cast<float>();
        for (const Eigen::Vector2f& point : pose.points_on_feature) {
          Eigen::Vector2f point_transformed = point_to_world * point;
          b_points.push_back(point_transformed);
        }
      }
      gui_helpers::AddLine(Eigen::Vector3f(hitl_constraint.line_a.start.x(),
                                           hitl_constraint.line_a.start.y(),
                                           0.0),
                           Eigen::Vector3f(hitl_constraint.line_a.end.x(),
                                           hitl_constraint.line_a.end.y(),
                                           0.0),
                           gui_helpers::Color4f::kMagenta,
                           &line_marker);
      for (int i = 0; i < 5; i++) {
        pointcloud_helpers::PublishPointcloud(
            line_a_poses, pose_a_point_marker, constraint_a_pose_pub);
        pointcloud_helpers::PublishPointcloud(
            line_b_poses, pose_b_point_marker, constraint_b_pose_pub);
        pointcloud_helpers::PublishPointcloud(
            a_points, a_points_marker, point_a_pub);
        pointcloud_helpers::PublishPointcloud(
            b_points, b_points_marker, point_b_pub);
        line_pub.publish(line_marker);
      }
    }
    PubConstraintPointclouds();
    PubKeyframes();
  }

  void PubKeyframes() {
    std::vector<Eigen::Vector2f> poses;
    for (const ds::LearnedKeyframe& frame : keyframes) {
      // std::cout << "Frame #: " << frame.node_idx << std::endl;
      const double* pose_arr = (*solution)[frame.node_idx].pose;
      Eigen::Vector2f pose_point(pose_arr[0], pose_arr[1]);
      poses.push_back(pose_point);
    }
    pointcloud_helpers::PublishPointcloud(poses, keyframe_marker, keyframe_pub);
  }

  void AddPosePointcloud(std::vector<Eigen::Vector2f>& pointcloud,
                         const ds::LCPose& pose) {
    size_t node_idx = pose.node_idx;
    std::vector<slam_types::SLAMNodeSolution2D> solution_c = *solution;
    std::vector<Eigen::Vector2f> p_cloud =
        problem.nodes[node_idx].lidar_factor.pointcloud;
    double* pose_arr = solution_c[node_idx].pose;
    Eigen::Affine2f robot_to_world =
        PoseArrayToAffine(&pose_arr[2], &pose_arr[0]).cast<float>();
    for (const Eigen::Vector2f& p : p_cloud) {
      pointcloud.push_back(robot_to_world * p);
    }
  }

  void PubConstraintPointclouds() {
    std::vector<Eigen::Vector2f> pointclouds;
    for (const auto& hitl_constraint : hitl_constraints) {
      for (const ds::LCPose& pose : hitl_constraint.line_a_poses) {
        AddPosePointcloud(pointclouds, pose);
      }
      for (const ds::LCPose& pose : hitl_constraint.line_b_poses) {
        AddPosePointcloud(pointclouds, pose);
      }
    }
    pointcloud_helpers::PublishPointcloud(
        pointclouds, hitl_points_marker, hitl_pointclouds);
  }

  void ClearNormals() { gui_helpers::ClearMarker(&normals_marker); }

  void AddConstraint(const ds::HitlLCConstraint& constraint) {
    hitl_constraints.push_back(constraint);
  }
  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    PubVisualization();
    return ceres::SOLVER_CONTINUE;
  }

  void UpdateProblemAndSolution(
      slam_types::SLAMNode2D& new_node,
      std::vector<slam_types::SLAMNodeSolution2D>* new_solution,
      slam_types::OdometryFactor2D& new_odom_factor) {
    CHECK_EQ(new_node.node_idx,
             (*new_solution)[new_solution->size() - 1].node_idx);
    CHECK_EQ(new_node.node_idx, new_odom_factor.pose_j);
    problem.nodes.push_back(new_node);
    problem.odometry_factors.push_back(new_odom_factor);
    solution = new_solution;
    CHECK_EQ(solution->size(), problem.nodes.size());
  }

  void UpdateProblemAndSolution(
      slam_types::SLAMNode2D& new_node,
      std::vector<slam_types::SLAMNodeSolution2D>* new_solution) {
    CHECK_EQ(new_node.node_idx,
             (*new_solution)[new_solution->size() - 1].node_idx);
    problem.nodes.push_back(new_node);
    solution = new_solution;
    CHECK_EQ(solution->size(), problem.nodes.size());
  }

 private:
  sensor_msgs::PointCloud2 all_points_marker;
  sensor_msgs::PointCloud2 new_points_marker;
  std::vector<Eigen::Vector2f> all_points;
  slam_types::SLAMProblem2D problem;
  std::vector<slam_types::SLAMNodeSolution2D>* solution;
  std::vector<ds::LearnedKeyframe>& keyframes;
  ros::Publisher point_pub;
  ros::Publisher pose_pub;
  ros::Publisher match_pub;
  ros::Publisher new_point_pub;
  ros::Publisher normals_pub;
  ros::Publisher constraint_a_pose_pub;
  ros::Publisher constraint_b_pose_pub;
  ros::Publisher keyframe_poses_pub;
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
  std::vector<ds::HitlLCConstraint> hitl_constraints;
  std::vector<ds::AutoLCConstraint> auto_constraints;
};
}  // namespace nautilus