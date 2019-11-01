//
// Created by jack on 9/25/19.
//

#include "./solver.h"

#include <omp.h>
#include <utility>
#include <queue>
#include <algorithm>
#include <queue>
#include <vector>

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"

#include "./kdtree.h"
#include "./slam_types.h"
#include "./math_util.h"
#include "./pointcloud_helpers.h"
#include "./gui_helpers.h"

#define LIDAR_CONSTRAINT_AMOUNT 10
#define OUTLIER_THRESHOLD 0.25

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
  Affine2T affine = Translation2T(translation[0], translation[1]) *
    Rotation2DT(rotation[0]).toRotationMatrix();
  return affine;
}

struct OdometryResidual {
    template <typename T>
    bool operator() (const T* pose_i,
                     const T* pose_j,
                     T* residual) const {
      // Predicted pose_j = pose_i * odometry.
      // Hence, error = pose_j.inverse() * pose_i * odometry;
      // Also A.inverse() = A.transpose() for rotation matrices because they
      // are orthogonal.
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
      residual[0] = T(translation_weight) * error_translation.x();
      residual[1] = T(translation_weight) * error_translation.y();
      residual[2] = T(rotation_weight) *
        Eigen::Rotation2D<T>().fromRotationMatrix(error_rotation_mat).angle();
      return true;
    }

    explicit OdometryResidual(const OdometryFactor2D& factor, double translation_weight, double rotation_weight) :
            translation_weight(translation_weight),
            rotation_weight(rotation_weight),
            R_odom(Eigen::Rotation2D<float>(factor.rotation)
                    .toRotationMatrix()),
            T_odom(factor.translation) {}

    static AutoDiffCostFunction<OdometryResidual, 3, 3, 3>* create(
            const OdometryFactor2D& factor,
            double translation_weight,
            double rotation_weight) {
      OdometryResidual* residual = new OdometryResidual(factor, translation_weight, rotation_weight);
      return new AutoDiffCostFunction<OdometryResidual, 3, 3, 3>(residual);
    }

    double translation_weight;
    double rotation_weight;
    const Matrix2f R_odom;
    const Vector2f T_odom;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

struct LIDARPointBlobResidual {
    template <typename T>
    bool operator() (const T* source_pose,
                     const T* target_pose,
                     T* residuals) const {
      typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      Affine2T source_to_world = PoseArrayToAffine(&source_pose[2], &source_pose[0]);
      Affine2T world_to_target = PoseArrayToAffine(&target_pose[2], &target_pose[0]).inverse();
      Affine2T source_to_target = world_to_target * source_to_world;
      #pragma omp parallel for
      for (size_t index = 0; index < source_points.size(); index++) {
        Vector2T source_pointT = source_points[index].cast<T>();
        Vector2T target_pointT = target_points[index].cast<T>();
        // Transform source_point into the frame of target_point
        source_pointT = source_to_target * source_pointT;
        T result = target_normals[index].cast<T>().dot(source_pointT - target_pointT);
        residuals[index] = result;
      }
      return true;
    }

    LIDARPointBlobResidual(vector<Vector2f>& source_points,
                           vector<Vector2f>& target_points,
                           vector<Vector2f>& target_normals) :
                           source_points(source_points),
                           target_points(target_points),
                           target_normals(target_normals) {
      CHECK_EQ(source_points.size(), target_points.size());
      CHECK_EQ(target_points.size(), target_normals.size());
    }

    static AutoDiffCostFunction<LIDARPointBlobResidual, ceres::DYNAMIC, 3, 3>*
    create(vector<Vector2f>& source_points,
           vector<Vector2f>& target_points,
           vector<Vector2f>& target_normals) {
      LIDARPointBlobResidual *residual =
        new LIDARPointBlobResidual(source_points,
                                    target_points,
                                    target_normals);
      return new AutoDiffCostFunction<LIDARPointBlobResidual,
                                      ceres::DYNAMIC,
                                      3, 3>
                                      (residual, source_points.size());
    }

    const vector<Vector2f> source_points;
    const vector<Vector2f> target_points;
    const vector<Vector2f> target_normals;
};

class VisualizationCallback : public ceres::IterationCallback {
 public:
  VisualizationCallback(const slam_types::SLAMProblem2D& problem,
                        const vector<slam_types::SLAMNodeSolution2D>* solution,
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
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_STRIP,
                                  gui_helpers::Color4f::kGreen,
                                  0.05,
                                  0.0,
                                  0.0,
                                  &pose_array);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kBlue,
                                  0.003,
                                  0.0,
                                  0.0,
                                  &match_line_list);
  }

  void PubVisualization() {
    const vector<slam_types::SLAMNodeSolution2D>& solution_c = *solution;
    vector<Vector2f> new_points;
    for (size_t i = 0; i < solution_c.size(); i++) {
      if (new_points.size() > 0) {
        all_points.insert(all_points.end(),
                          new_points.begin(),
                          new_points.end());
              new_points.clear();
      }
      auto pointcloud = problem.nodes[i].lidar_factor.pointcloud;
      Eigen::Affine2f robot_to_world =
        PoseArrayToAffine(&(solution_c[i].pose[2]),
                          &(solution_c[i].pose[0])).cast<float>();
      Eigen::Vector3f pose(solution_c[i].pose[0], solution_c[i].pose[1], 0.0);
      gui_helpers::AddPoint(pose, gui_helpers::Color4f::kGreen, &pose_array);
      for (const Vector2f& point : pointcloud) {
              new_points.push_back(robot_to_world * point);
      }
    }
    if (solution_c.size() >= 2) {
      pointcloud_helpers::PublishPointcloud(all_points,
                                            all_points_marker,
                                            point_pub);
      pointcloud_helpers::PublishPointcloud(new_points,
                                            new_points_marker,
                                            new_point_pub);
      pose_pub.publish(pose_array);
      match_pub.publish(match_line_list);
    }
    ros::spinOnce();
    all_points.clear();
    gui_helpers::ClearMarker(&pose_array);
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
  const slam_types::SLAMProblem2D& problem;
  const vector<slam_types::SLAMNodeSolution2D>* solution;
  ros::Publisher point_pub;
  ros::Publisher pose_pub;
  ros::Publisher match_pub;
  ros::Publisher new_point_pub;
  visualization_msgs::Marker pose_array;
};

void Solver::AddOdomFactors(const vector<OdometryFactor2D>& odom_factors,
                            vector<slam_types::SLAMNodeSolution2D>& solution,
                            ceres::Problem* ceres_problem) {
  for (const OdometryFactor2D& odom_factor : odom_factors) {
    CHECK_LT(odom_factor.pose_i, odom_factor.pose_j);
    CHECK_GT(solution.size(), odom_factor.pose_i);
    CHECK_GT(solution.size(), odom_factor.pose_j);
    ceres_problem->AddResidualBlock(
      OdometryResidual::create(odom_factor,
                               translation_weight,
                               rotation_weight),
      NULL,
      solution[odom_factor.pose_i].pose,
      solution[odom_factor.pose_j].pose);
  }
}

bool NormalsWithin(Vector2f& normal_1, Vector2f& normal_2, double threshold) {
  return acos(normal_1.dot(normal_2) / (normal_1.norm() * normal_2.norm())) < threshold;
}

// Source moves to target.
double
Solver::GetPointCorrespondences(const slam_types::SLAMProblem2D& problem,
                                vector<slam_types::SLAMNodeSolution2D>*
                                  solution_ptr,
                                PointCorrespondences* point_correspondences,
                                size_t source_node_index,
                                size_t target_node_index) {
  gui_helpers::ClearMarker(&match_line_list);
  // Get all the data we might need in an easier to use format.
  double difference = 0.0;
  vector<SLAMNodeSolution2D>& solution = *solution_ptr;
  SLAMNodeSolution2D& source_solution = solution[source_node_index];
  SLAMNodeSolution2D& target_solution = solution[target_node_index];
  LidarFactor source_lidar =
    problem.nodes[source_node_index].lidar_factor;
  LidarFactor target_lidar =
    problem.nodes[target_node_index].lidar_factor;
  Eigen::Affine2f source_to_world =
    PoseArrayToAffine(&source_solution.pose[2],
                      &source_solution.pose[0]).cast<float>();
  Eigen::Affine2f target_to_world =
    PoseArrayToAffine(&target_solution.pose[2],
                      &target_solution.pose[0]).cast<float>();
  // Create a point correspondence to track the closest points.
  for (const Vector2f& source_point : source_lidar.pointcloud) {
    // Transform the source point to the target frame.
    Vector2f source_point_transformed =
      target_to_world.inverse() * source_to_world * source_point;
    // Get the closest point and the closest surface normal.
    KDNodeValue<float, 2> closest_target;
    vector<KDNodeValue<float, 2>> neighbors;
    target_lidar.pointcloud_tree->FindNeighborPoints(source_point_transformed,
                                                     OUTLIER_THRESHOLD,
                                                     &neighbors);
    KDNodeValue<float, 2> source_point_with_normal;
    float found_dist =
      source_lidar.pointcloud_tree->FindNearestPoint(source_point,
                                                     OUTLIER_THRESHOLD,
                                                     &source_point_with_normal);
    if (found_dist != 0.0) {
      std::cerr << "Something is wrong, this point is not in its KDTree!" << std::endl;
      exit(1);
    }
    float dist = OUTLIER_THRESHOLD;
    // Sort the neighbor points so we will always get the smallest distance, and best
    // normal.
    std::sort(neighbors.begin(),
              neighbors.end(),
              [&source_point_transformed](KDNodeValue<float, 2> point_1,
                                          KDNodeValue<float, 2> point_2) {
      return (source_point_transformed - point_1.point).norm() <
             (source_point_transformed - point_2.point).norm();
    });
    for (KDNodeValue<float, 2> current_target : neighbors) {
      if (NormalsWithin(current_target.normal,
                        source_point_with_normal.normal,
                        M_PI / 9)) {
        closest_target = current_target;
        dist = (source_point_transformed - current_target.point).norm();
        break;
      }
    }
    // Outlier rejection
    if (dist >= OUTLIER_THRESHOLD) {
      // No point was found within our given threshold.
      continue;
    }
    difference += dist;
    // Minimize distance between closest points!
    Vector2f source_point_modifiable = source_point;
    point_correspondences->source_points.push_back(source_point_modifiable);
    point_correspondences->target_points.push_back(closest_target.point);
    point_correspondences->target_normals.push_back(closest_target.normal);
    // Add a line from the matches that we are using.
    // Transform everything to the world frame.
    // This is for the visualization.
    source_point_transformed = target_to_world * source_point_transformed;
    Vector2f closest_point_in_target = target_to_world * closest_target.point;
    Eigen::Vector3f source_3d(source_point_transformed.x(),
                              source_point_transformed.y(), 0.0f);
    Eigen::Vector3f target_3d(closest_point_in_target.x(),
                              closest_point_in_target.y(), 0.0f);
    gui_helpers::AddLine(source_3d,
                         target_3d,
                         gui_helpers::Color4f::kBlue,
                         &match_line_list);
  }
  return difference;
}

vector<slam_types::SLAMNodeSolution2D>
Solver::SolveSLAM(slam_types::SLAMProblem2D& problem,
                       ros::NodeHandle& n) {
  // Copy all the data to a list that we are going to modify as we optimize.
  vector<slam_types::SLAMNodeSolution2D> solution(problem.nodes.size());
  for (size_t i = 0; i < problem.nodes.size(); i++) {
    // Make sure that we marked all the data correctly earlier.
    CHECK_EQ(i, problem.nodes[i].node_idx);
    SLAMNodeSolution2D sol_node(problem.nodes[i]);
    solution[i] = sol_node;
  }
  // Setup ceres for evaluation of the problem.
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  VisualizationCallback vis_callback(problem, &solution, n);
  options.callbacks.push_back(&vis_callback);
  // Sliding window for solving for initial guesses.
  double difference = 0;
  double last_difference = 0;
  do {
    last_difference = difference;
    difference = 0;
    ceres::Problem ceres_problem;
    AddOdomFactors(problem.odometry_factors, solution, &ceres_problem);
    for (size_t node_i_index = 0;
        node_i_index < problem.nodes.size();
        node_i_index++) {
      // Set the first pose to be constant.
      if (node_i_index == 0) {
        ceres_problem.SetParameterBlockConstant(solution[0].pose);
      }
      for (size_t node_j_index =
             std::max((int64_t)(node_i_index) - LIDAR_CONSTRAINT_AMOUNT, 0l);
           node_j_index < node_i_index;
           node_j_index++) {
        // Add all the points to this, make a new problem. Minimize, continue.
        PointCorrespondences correspondence(solution[node_j_index].pose,
                                            solution[node_i_index].pose);
        difference += GetPointCorrespondences(problem,
                                              &solution,
                                              &correspondence,
                                              node_j_index,
                                              node_i_index);
        ceres_problem.AddResidualBlock(
          LIDARPointBlobResidual::create(correspondence.source_points,
                                         correspondence.target_points,
                                         correspondence.target_normals),
          NULL,
          correspondence.source_pose,
          correspondence.target_pose);
      }
    }
    ceres::Solve(options, &ceres_problem, &summary);
  } while (abs(difference - last_difference) > stopping_accuracy);
  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i <  5; i++) {
    vis_callback.PubVisualization();
    sleep(1);
  }
  return solution;
}

Solver::Solver(double translation_weight,
               double rotation_weight,
               double stopping_accuracy) :
               translation_weight(translation_weight),
               rotation_weight(rotation_weight),
               stopping_accuracy(stopping_accuracy) {}

