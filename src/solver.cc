//
// Created by jack on 9/25/19.
//

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"

#include "solver.h"

#include <utility>
#include <queue>
#include <algorithm>
#include "valgrind/memcheck.h"
#include "slam_types.h"
#include "math_util.h"
#include "pointcloud_helpers.h"
#include "gui_helpers.h"
#include <queue>
#include "kdtree.h"

#define LIDAR_CONSTRAINT_AMOUNT 10
#define OUTLIER_THRESHOLD 0.80

using std::vector;
using slam_types::OdometryFactor2D;
using slam_types::LidarFactor;
using ceres::AutoDiffCostFunction;
using Eigen::Matrix2f;
using Eigen::Vector2f;
using slam_types::SLAMNodeSolution2D;

visualization_msgs::Marker match_line_list;

template<typename T> Eigen::Transform<T, 2, Eigen::Affine>
PoseArrayToAffine(const T* rotation, const T* translation) {
  typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
  typedef Eigen::Rotation2D<T> Rotation2DT;
  typedef Eigen::Translation<T, 2> Translation2T;
  Affine2T affine = Translation2T(translation[0], translation[1]) * Rotation2DT(rotation[0]).toRotationMatrix();
  return affine;
}

struct OdometryResidual {
    template <typename T>
    bool operator() (const T* pose_i,
                     const T* pose_j,
                     T* residual) const {
      // Predicted pose_j = pose_i * odometry.
      // Hence, error = pose_j.inverse() * pose_i * odometry;
      // Also A.inverse() = A.transpose() for rotation matrices because they are
      // orthogonal.
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      typedef Eigen::Matrix<T, 2, 2> Matrix2T;

      const Matrix2T Ri = Eigen::Rotation2D<T>(pose_i[2])
              .toRotationMatrix();
      const Matrix2T Rj = Eigen::Rotation2D<T>(pose_j[2])
              .toRotationMatrix();

      const Vector2T Ti(pose_i[0], pose_i[1]);
      const Vector2T Tj(pose_j[0], pose_j[1]);

      const Vector2T error_translation =
              Rj.transpose() * (Ri * T_odom.cast<T>() - (Tj - Ti));

      const Matrix2T error_rotation_mat =
              Rj.transpose() * Ri * R_odom.cast<T>();
      residual[0] = error_translation.x();
      residual[1] = error_translation.y();
      residual[2] = Eigen::Rotation2D<T>().fromRotationMatrix(error_rotation_mat).angle();
      return true;
    }

    explicit OdometryResidual(const OdometryFactor2D& factor) :
            R_odom(Eigen::Rotation2D<float>(factor.rotation)
                    .toRotationMatrix()),
            T_odom(factor.translation) {}

    static AutoDiffCostFunction<OdometryResidual, 3, 3, 3>* create(
            const OdometryFactor2D& factor) {
      OdometryResidual* residual = new OdometryResidual(factor);
      return new AutoDiffCostFunction<OdometryResidual, 3, 3, 3>(residual);
    }

    const Matrix2f R_odom;
    const Vector2f T_odom;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

struct LIDARPointResidual {
    template <typename T>
    bool operator() (const T* source_pose,
                     const T* target_pose,
                     T* residuals) const {
      typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      Affine2T source_to_world = PoseArrayToAffine(&source_pose[2], &source_pose[0]);
      Affine2T target_to_world = PoseArrayToAffine(&target_pose[2], &target_pose[0]);
      Vector2T source_pointT = source_point.cast<T>();
      Vector2T target_pointT = target_point.cast<T>();
      // Transform source_point into the frame of target_point
      source_pointT = target_to_world.inverse() * source_to_world * source_pointT;
      T result = target_normal.cast<T>().dot(source_pointT - target_pointT);
      residuals[0] = result;
      //residuals[1] = result.y();
      return true;
    }

    LIDARPointResidual(Vector2f& source_point,
                       Vector2f& target_point,
                       Vector2f& target_normal) :
            source_point(source_point),
            target_point(target_point),
            target_normal(target_normal) {}

    static AutoDiffCostFunction<LIDARPointResidual, 1, 3, 3>* create(
            Vector2f& source_point, Vector2f& target_point, Vector2f& target_normal) {
      LIDARPointResidual *residual = new LIDARPointResidual(source_point, target_point, target_normal);
      return new AutoDiffCostFunction<LIDARPointResidual, 1, 3, 3>(residual);
    }

    const Vector2f source_point;
    const Vector2f target_point;
    const Vector2f target_normal;
};

class VisualizationCallback : public ceres::IterationCallback {
public:
    VisualizationCallback(const slam_types::SLAMProblem2D& problem,
            const vector<slam_types::SLAMNodeSolution2D>* solution,
            ros::NodeHandle& n) : problem(problem), solution(solution) {
      pointcloud_helpers::InitPointcloud(&all_points_marker);
      pointcloud_helpers::InitPointcloud(&new_points_marker);
      all_points.clear();
      point_pub = n.advertise<sensor_msgs::PointCloud2>("/all_points", 10);
      new_point_pub = n.advertise<sensor_msgs::PointCloud2>("/new_points", 10);
      pose_pub = n.advertise<visualization_msgs::Marker>("/poses", 10);
      match_pub = n.advertise<visualization_msgs::Marker>("/matches", 10);
      gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_STRIP, gui_helpers::Color4f::kGreen, 0.05, 0.0, 0.0, &pose_array);
      gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST, gui_helpers::Color4f::kBlue, 0.01, 0.0, 0.0, &match_line_list);
    }

    void PubVisualization() {
      const vector<slam_types::SLAMNodeSolution2D>& solution_c = *solution;
      vector<Vector2f> new_points;
      for (size_t i = 0; i < solution_c.size() && solution_c[i].visualize; i++) {
        if (new_points.size() > 0) {
          all_points.insert(all_points.end(), new_points.begin(), new_points.end());
                new_points.clear();
        }
        auto pointcloud = problem.nodes[i].lidar_factor.pointcloud;
        (void) pointcloud;
        Eigen::Affine2f robot_to_world = PoseArrayToAffine(&(solution_c[i].pose[2]), &(solution_c[i].pose[0])).cast<float>();
        Eigen::Vector3f pose(solution_c[i].pose[0], solution_c[i].pose[1], 0.0);
        gui_helpers::AddPoint(pose, gui_helpers::Color4f::kGreen, &pose_array);
        for (const Vector2f& point : pointcloud) {
                new_points.push_back(robot_to_world * point);
        }
      }
      pointcloud_helpers::PublishPointcloud(all_points, all_points_marker, point_pub);
      pointcloud_helpers::PublishPointcloud(new_points, new_points_marker, new_point_pub);
      pose_pub.publish(pose_array);
      match_pub.publish(match_line_list);
      ros::spinOnce();
      all_points.clear();
      gui_helpers::ClearMarker(&pose_array);
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
      PubVisualization();
      return ceres::SOLVER_CONTINUE;
    }

private:
    sensor_msgs::PointCloud2 all_points_marker;
    sensor_msgs::PointCloud2 new_points_marker;
    std::vector<Vector2f> all_points;
    const slam_types::SLAMProblem2D& problem;
    const vector<slam_types::SLAMNodeSolution2D>* solution;
    ros::Publisher point_pub;
    ros::Publisher pose_pub;
    ros::Publisher match_pub;
    ros::Publisher new_point_pub;
    visualization_msgs::Marker pose_array;
};

void AddOdomFactors(const slam_types::SLAMProblem2D& problem,
                    vector<slam_types::SLAMNodeSolution2D>& solution,
                    ceres::Problem* ceres_problem) {

  for (const OdometryFactor2D& odom_factor : problem.odometry_factors) {
    CHECK_LT(odom_factor.pose_i, odom_factor.pose_j);
    CHECK_GT(solution.size(), odom_factor.pose_i);
    CHECK_GT(solution.size(), odom_factor.pose_j);
    ceres_problem->AddResidualBlock(OdometryResidual::create(odom_factor),
                                    NULL,
                                    solution[odom_factor.pose_i].pose,
                                    solution[odom_factor.pose_j].pose);
  }
}

struct LidarPointMatch {
  Vector2f source_point;
  Vector2f target_point;
  Vector2f target_normal;
  double *source_pose;
  double *target_pose;
  
  LidarPointMatch(Vector2f source_point, Vector2f target_point, Vector2f target_normal, double* source_pose, double* target_pose) :
                  source_point(source_point), target_point(target_point), target_normal(target_normal), source_pose(source_pose), target_pose(target_pose) {}
  
};

// Source moves to target.
double GetClosestTargetPointAndNormal(const slam_types::SLAMProblem2D& problem,
                              vector<slam_types::SLAMNodeSolution2D>* solution_ptr,
                              vector<LidarPointMatch>* matches,
                              size_t source_node_index,
                              size_t target_node_index) {
  gui_helpers::ClearMarker(&match_line_list);
  // Loop over the two point clouds finding the closest points.
  double difference = 0.0;
  vector<SLAMNodeSolution2D>& solution = *solution_ptr;
  SLAMNodeSolution2D& source_solution = solution[source_node_index];
  SLAMNodeSolution2D& target_solution = solution[target_node_index];
  const vector<Vector2f>& source_pointcloud =
          problem.nodes[source_node_index].lidar_factor.pointcloud;
  LidarFactor target_lidar =
          problem.nodes[target_node_index].lidar_factor;
  Eigen::Affine2f source_to_world =
          PoseArrayToAffine(&source_solution.pose[2], &source_solution.pose[0]).cast<float>();
  Eigen::Affine2f target_to_world =
          PoseArrayToAffine(&target_solution.pose[2], &target_solution.pose[0]).cast<float>();
  for (const Vector2f& source_point : source_pointcloud) {
    // Transform the source point to the target frame.
    Vector2f source_point_transformed =
            target_to_world.inverse() * source_to_world * source_point;
    // Get the closest point and the closest surface normal.
    KDNodeValue<float, 2> closest_target;
    float dist = target_lidar.pointcloud_tree->FindNearestPoint(source_point_transformed,
                                                                OUTLIER_THRESHOLD,
                                                                &closest_target);
    // Outlier rejection
    if (dist >= OUTLIER_THRESHOLD) {
      // No point was found within our given threshold.
      continue;
    }
    difference += dist;
    // Minimize distance between closest points!
    Vector2f source_point_modifiable = source_point;
    matches->emplace_back(source_point_modifiable,
                          closest_target.point,
                          closest_target.normal,
                          source_solution.pose,
                          target_solution.pose);
    // Add a line from the matches that we are using.
    // Transform everything to the world frame.
    source_point_transformed = target_to_world * source_point_transformed;
    Vector2f closest_point_in_target = target_to_world * closest_target.point;
    Eigen::Vector3f source_3d(source_point_transformed.x(), source_point_transformed.y(), 0.0f);
    Eigen::Vector3f target_3d(closest_point_in_target.x(), closest_point_in_target.y(), 0.0f);
    gui_helpers::AddLine(source_3d, target_3d, gui_helpers::Color4f::kBlue, &match_line_list);
  }
  return difference;
}

void AddLidarMatchResiduals(ceres::Problem* ceres_problem, vector<LidarPointMatch>& matches) {
  for (LidarPointMatch& match : matches) {
    ceres_problem->AddResidualBlock(LIDARPointResidual::create(match.source_point, match.target_point, match.target_normal),
              NULL,
              match.source_pose,
              match.target_pose);
  }
}

bool solver::SolveSLAM(slam_types::SLAMProblem2D& problem, ros::NodeHandle& n) {
  // Copy all the data to a list that we are going to modify as we optimize.
  vector<slam_types::SLAMNodeSolution2D> solution(problem.nodes.size());
  for (size_t i = 0; i < problem.nodes.size(); i++) {
    // Make sure that we marked all the data correctly earlier.
    CHECK_EQ(i, problem.nodes[i].node_idx);
    SLAMNodeSolution2D sol_node(problem.nodes[i]);
    solution[i] = sol_node;
  }
  // Setup ceres for evaluation of the problem.
  ceres::Problem global_problem;
  ceres::Solver::Options global_options;
  ceres::Solver::Options local_options;
  ceres::Solver::Summary summary;
  global_options.linear_solver_type = ceres::DENSE_QR;
  global_options.minimizer_progress_to_stdout = false;
  global_options.num_threads = 4;
  local_options.linear_solver_type = ceres::DENSE_QR;
  local_options.minimizer_progress_to_stdout = false;
  local_options.num_threads = 4;
  /* 
   * Add the visualization to the global problem.
   * We don't want to run it on each iteration of the local problem
   * because that will slow things down.
   */
  VisualizationCallback vis_callback(problem, &solution, n);
  //global_options.callbacks.push_back(&vis_callback);
  AddOdomFactors(problem, solution, &global_problem);
  for (size_t node_i_index = 0;
       node_i_index < problem.nodes.size();
       node_i_index++) {
    double difference = 0;
    double last_difference = 0;
    solution[node_i_index].visualize = true;
    vector<LidarPointMatch> current_matches;
    do {
      current_matches.clear();
      last_difference = difference;
      difference = 0;
      ceres::Problem local_problem;
      AddOdomFactors(problem, solution, &local_problem);
      for (size_t node_j_index = std::max((long)(node_i_index) - LIDAR_CONSTRAINT_AMOUNT, 0l);
           node_j_index < node_i_index;
           node_j_index++) {
        // Add all the points to this, make a new problem. Minimize, continue.
        difference += GetClosestTargetPointAndNormal(problem,
                                            &solution,
                                            &current_matches,
                                            node_j_index,
                                            node_i_index);
      }
      AddLidarMatchResiduals(&local_problem, current_matches);
      ceres::Solve(local_options, &local_problem, &summary);
    } while(abs(difference - last_difference) > 0.05);
    AddLidarMatchResiduals(&global_problem, current_matches);
    ceres::Solve(global_options, &global_problem, &summary);
    // Sometimes ceres doesn't actually call the visualization.
    vis_callback.PubVisualization();
    printf("Solved for a lidar node.\n");
  }
  // Call the visualization once more to see the finished optimization.
  vis_callback.PubVisualization();
  return true;
}

