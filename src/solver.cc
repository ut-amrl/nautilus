//
// Created by jack on 9/25/19.
//

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"

#include "solver.h"
#include "slam_types.h"
#include "math_util.h"
#include "pointcloud_helpers.h"

using std::vector;
using slam_types::OdometryFactor2D;
using slam_types::LidarFactor;
using ceres::AutoDiffCostFunction;
using Eigen::Matrix2f;
using Eigen::Vector2f;


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

      T rotation_error =
              atan2(error_rotation_mat(0, 1), error_rotation_mat(0, 0));

      residual[0] = error_translation[0];
      residual[1] = error_translation[1];
      residual[2] = rotation_error;
      return true;
    }

    OdometryResidual(const OdometryFactor2D& factor) :
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

#define LIDAR_THRESHOLD 0.15

struct LidarPointResidual {
    template <typename T>
    bool operator() (const T* transformation,
                     T* residuals) const {
      // Apply the transformation and find the closest point in the other list.
      // If the closest point is still further then our threshold then
      // set this residual to zero.
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
      // Make sure we have a lidar scan to compare against.
      CHECK_GT(lidar_j.size(), 0);
      // Make a rotation matrix.
      Vector2T point_t = point_i.cast<T>();
      Affine2T i_to_world;
      i_to_world.rotate(transformation[2])
           .translate(Vector2T(transformation[0], transformation[1]));
      point_t = i_to_world * point_t;
      CHECK(ceres::IsFinite(point_t.x()) && ceres::IsFinite(point_t.y()));
      Vector2T closest_point = lidar_j[0].cast<T>();
      for (const Vector2f vec : lidar_j) {
        Vector2T vec_t = vec.cast<T>();
        CHECK(ceres::IsFinite(vec_t.x()) && ceres::IsFinite(vec_t.y()));
        if ((vec_t - point_t).isZero(T(0.001)) || (vec_t - point_t).norm() <
            (closest_point - point_t).norm()) {
          closest_point = vec_t;
        }
      }
      T count_point = T(1.0);
      if (!(closest_point - point_t).isZero(T(0.001)) && (closest_point - point_t).norm() > LIDAR_THRESHOLD) {
        count_point = T(0.0);
      }
      CHECK(ceres::IsFinite(count_point * (point_t.x() - closest_point.x())));
      CHECK(ceres::IsFinite(count_point * (point_t.y() - closest_point.y())));
      residuals[0] = count_point * (point_t.x() - closest_point.x());
      residuals[1] = count_point * (point_t.y() - closest_point.y());
      return true;
    }

    LidarPointResidual(const Vector2f& point_i,
                       const std::vector<Vector2f>& lidar_factor_j) :
                  point_i(point_i),
                  lidar_j(lidar_factor_j) {}

    static AutoDiffCostFunction<LidarPointResidual, 2, 3>* create(
            const Vector2f& point_i,
            const slam_types::SLAMNode2D& factor_j) {
      auto lidar_factor_j = factor_j.lidar_factor.pointcloud;
      // Want to transform all lidar_factor points to the reference frame
      // of point_i, but to do this we first have to transform it by A^2_W_
      // then inside the residual by A^W_1_
      for (Vector2f vec : lidar_factor_j) {
        Eigen::Affine2f affine;
        affine.rotate(factor_j.pose.angle).translate(factor_j.pose.loc);
        vec = affine * vec;
      }
      LidarPointResidual* residual = new LidarPointResidual(point_i,
                                                            lidar_factor_j);
      return new AutoDiffCostFunction<LidarPointResidual, 2, 3>(residual);
    }

    const Vector2f point_i;
    const vector<Vector2f>& lidar_j;
};

#define LIDAR_CONSTRAINT_AMOUNT 10

void
AddLidarResiduals(const slam_types::SLAMProblem2D& problem,
                  std::vector<slam_types::SLAMNodeSolution2D>& solution,
                  ceres::Problem* ceres_problem) {
  CHECK_LE(LIDAR_CONSTRAINT_AMOUNT, problem.nodes.size());
  for (size_t node_i_index = LIDAR_CONSTRAINT_AMOUNT;
       node_i_index < problem.nodes.size();
       node_i_index++) {
    for (size_t node_j_index = node_i_index - LIDAR_CONSTRAINT_AMOUNT;
         node_j_index < node_i_index;
         node_j_index++) {
      auto* pose_i_block =
              reinterpret_cast<double *>(&solution[node_i_index].pose);
      for (const Vector2f& point :
           problem.nodes[node_i_index].lidar_factor.pointcloud) {
        ceres_problem->AddResidualBlock(
                LidarPointResidual::create(point,
                        problem.nodes[node_j_index]),
                        NULL,
                        pose_i_block);
      }
    }
  }
}

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
  ceres::Problem ceres_problem;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  for (const OdometryFactor2D& odom_factor : problem.odometry_factors) {
    // Add all the odometry residuals. These will act as a constraint to keep
    // the optimizer from going crazy.
    auto* pose_i_block =
            reinterpret_cast<double *>(&solution[odom_factor.pose_i].pose);
    auto* pose_j_block =
            reinterpret_cast<double *>(&solution[odom_factor.pose_j].pose);
    ceres_problem.AddResidualBlock(OdometryResidual::create(odom_factor),
            NULL,
            pose_i_block,
            pose_j_block);
  }
  AddLidarResiduals(problem, solution, &ceres_problem);
//  VisualizationCallback vis_callback(problem, &solution, n);
//  options.callbacks.push_back(&vis_callback);
  ceres::Solve(options, &ceres_problem, &summary);
  printf("%s\n", summary.FullReport().c_str());

  return true;
}

