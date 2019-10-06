//
// Created by jack on 9/25/19.
//

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"

#include "solver.h"

#include <utility>
#include "slam_types.h"
#include "math_util.h"
#include "pointcloud_helpers.h"

using std::vector;
using slam_types::OdometryFactor2D;
using slam_types::LidarFactor;
using ceres::AutoDiffCostFunction;
using Eigen::Matrix2f;
using Eigen::Vector2f;
using slam_types::SLAMNodeSolution2D;

template<typename T> Eigen::Transform<T, 2, Eigen::Affine>
PoseArrayToAffine(const T* rotation, const T* translation) {
  typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
  typedef Eigen::Matrix<T, 2, 1> Vector2T;
  Affine2T affine;
  affine.rotate(rotation[0]).translate(Vector2T(translation[0], translation[1]));
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

      const Vector2T error_translation = Ti + T_odom.cast<T>() - Tj;

      const Matrix2T error_rotation_mat =
              Rj.transpose() * Ri * R_odom.cast<T>();

      residual[0] = error_translation[0];
      residual[1] = error_translation[1];
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
      CHECK(ceres::IsFinite(source_point.x()) && ceres::IsFinite(source_point.y()));
      CHECK(ceres::IsFinite(source_pose[0]) && ceres::IsFinite(source_pose[1]) && ceres::IsFinite(source_pose[2]));
      CHECK(ceres::IsFinite(target_pose[0]) && ceres::IsFinite(target_pose[1]) && ceres::IsFinite(target_pose[2]));
      for (int64_t row = 0; row < source_to_world.rows(); row++) {
        for (int64_t col = 0; col < source_to_world.cols(); col++) {
          CHECK(ceres::IsFinite(source_to_world(row, col)));
          CHECK(ceres::IsFinite(target_to_world.inverse()(row, col)));
        }
        printf("\n");
      }
      Vector2T target_pointT = target_point.cast<T>();
      // Transform source_point into the frame of target_point
      CHECK(ceres::IsFinite(source_pointT.x()) && ceres::IsFinite(source_pointT.y()));
      source_pointT = target_to_world.inverse() * source_to_world * source_pointT;
      CHECK(ceres::IsFinite(source_pointT.x()) && ceres::IsFinite(source_pointT.y()));
      residuals[0] = source_pointT.x() - target_pointT.x();
      residuals[1] = source_pointT.y() - target_pointT.y();
      return true;
    }

    LIDARPointResidual(Vector2f source_point,
                       Vector2f target_point) :
            source_point(std::move(source_point)),
            target_point(std::move(target_point)) {}

    static AutoDiffCostFunction<LIDARPointResidual, 2, 3, 3>* create(
            const Vector2f& source_point, const Vector2f& target_point) {
      LIDARPointResidual *residual = new LIDARPointResidual(source_point, target_point);
      return new AutoDiffCostFunction<LIDARPointResidual, 2, 3, 3>(residual);
    }

    const Vector2f source_point;
    const Vector2f target_point;
};

class VisualizationCallback : public ceres::IterationCallback {
public:
    VisualizationCallback(const slam_types::SLAMProblem2D& problem,
            const vector<slam_types::SLAMNodeSolution2D>* solution,
            ros::NodeHandle& n) : problem(problem), solution(solution) {
      pointcloud_helpers::InitPointcloud(&all_points_marker);
      all_points.clear();
      point_pub = n.advertise<sensor_msgs::PointCloud2>("/all_points", 10);
    }

    void PubVisualization() {
      const vector<slam_types::SLAMNodeSolution2D>& solution_c = *solution;
      for (size_t i = 0; i < solution_c.size(); i++) {
        auto pointcloud = problem.nodes[i].lidar_factor.pointcloud;
        (void) pointcloud;
        Eigen::Affine2f robot_to_world = PoseArrayToAffine(&(solution_c[i].pose[2]), &(solution_c[i].pose[0])).cast<float>();
        for (const Vector2f& point : pointcloud) {
          all_points.push_back(robot_to_world * point);
        }
      }
      pointcloud_helpers::PublishPointcloud(all_points, all_points_marker, point_pub);
      ros::spinOnce();
      all_points.clear();
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
      PubVisualization();
      return ceres::SOLVER_CONTINUE;
    }
private:
    sensor_msgs::PointCloud2 all_points_marker;
    std::vector<Vector2f> all_points;
    const slam_types::SLAMProblem2D& problem;
    const vector<slam_types::SLAMNodeSolution2D>* solution;
    ros::Publisher point_pub;
};

void AddOdomFactors(const slam_types::SLAMProblem2D& problem,
                    const vector<slam_types::SLAMNodeSolution2D>& solution,
                    ceres::Problem* ceres_problem) {
  for (const OdometryFactor2D& odom_factor : problem.odometry_factors) {
    double* pose_i_block =
            const_cast<double*>(solution[odom_factor.pose_i].pose);
    double* pose_j_block =
            const_cast<double*>(solution[odom_factor.pose_j].pose);
    ceres_problem->AddResidualBlock(OdometryResidual::create(odom_factor),
                                    NULL,
                                    pose_i_block,
                                    pose_j_block);
  }
}


#define LIDAR_CONSTRAINT_AMOUNT 10

// Source moves to target.
void AddPointCloudFactors(const slam_types::SLAMProblem2D& problem,
                          const vector<slam_types::SLAMNodeSolution2D>& solution,
                          ceres::Problem* ceres_problem,
                          size_t source_node_index,
                          size_t target_node_index) {
  // Loop over the two point clouds finding the closest points.
  const SLAMNodeSolution2D& source_solution = solution[source_node_index];
  const SLAMNodeSolution2D& target_solution = solution[target_node_index];
  const vector<Vector2f>& source_pointcloud =
          problem.nodes[source_node_index].lidar_factor.pointcloud;
  const vector<Vector2f>& target_pointcloud =
          problem.nodes[target_node_index].lidar_factor.pointcloud;
  Eigen::Affine2f source_to_world =
          PoseArrayToAffine(&source_solution.pose[2], &source_solution.pose[0]).cast<float>();
  Eigen::Affine2f target_to_world =
          PoseArrayToAffine(&target_solution.pose[2], &target_solution.pose[0]).cast<float>();
  for (const Vector2f& source_point : source_pointcloud) {
    Vector2f source_point_transformed =
            target_to_world.inverse() * source_to_world * source_point;
    float min_distance = MAXFLOAT;
    Vector2f closest_target = target_pointcloud[0];
    for(const Vector2f& target_point : target_pointcloud) {
      if ((target_point - source_point_transformed).norm() < min_distance) {
        closest_target = target_point;
        min_distance = (target_point - source_point_transformed).norm();
      }
    }
    // Minimize distance between closest points!
    ceres_problem->AddResidualBlock(LIDARPointResidual::create(source_point, closest_target),
            NULL,
            const_cast<double*>(source_solution.pose),
            const_cast<double*>(target_solution.pose));
  }
}

bool solver::SolveSLAM(slam_types::SLAMProblem2D& problem, ros::NodeHandle& n) {
  // Copy all the data to a list that we are going to modify as we optimize.
  vector<slam_types::SLAMNodeSolution2D> solution;
  for (size_t i = 0; i < problem.nodes.size(); i++) {
    // Make sure that we marked all the data correctly earlier.
    CHECK_EQ(i, problem.nodes[i].node_idx);
    solution.emplace_back(problem.nodes[i]);
  }
  // Setup ceres for evaluation of the problem.
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  // Add the visualization.
  VisualizationCallback vis_callback(problem, &solution, n);
  options.callbacks.push_back(&vis_callback);
  // Continually solve and minimize.
  for (size_t node_i_index = LIDAR_CONSTRAINT_AMOUNT;
       node_i_index < problem.nodes.size();
       node_i_index++) {
    for (size_t node_j_index = node_i_index - LIDAR_CONSTRAINT_AMOUNT;
         node_j_index < node_i_index;
         node_j_index++) {
      // Add all the points to this, make a new problem. Minimize, continue.
      ceres::Problem ceres_problem;
      AddOdomFactors(problem, solution, &ceres_problem);
      AddPointCloudFactors(problem, solution, &ceres_problem, node_j_index, node_i_index);
      ceres::Solve(options, &ceres_problem, &summary);
      printf("%s\n", summary.FullReport().c_str());
    }
  }
  return true;
}

