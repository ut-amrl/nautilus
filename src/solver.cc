//
// Created by jack on 9/25/19.
//

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"

#include "solver.h"
#include "slam_types.h"
#include "math_util.h"

using std::vector;
using slam_types::OdometryFactor2D;
using slam_types::LidarFactor;
using ceres::AutoDiffCostFunction;
using Eigen::Matrix2f;
using Eigen::Vector2f;

template<typename T> Eigen::Transform<T, 3, Eigen::Affine>
PoseArrayToAffine(const T* rotation, const T* translation) {
  typedef Eigen::Transform<T, 3, Eigen::Affine> Affine3T;
  typedef Eigen::Matrix<T, 3, 1> Vector3T;
  typedef Eigen::AngleAxis<T> AngleAxisT;
  typedef Eigen::Translation<T, 3> Translation3T;

  const Vector3T rotation_axis(rotation[0], rotation[1], rotation[2]);
  const T rotation_angle = rotation_axis.norm();

  AngleAxisT rotation_aa(rotation_angle, rotation_axis / rotation_angle);
  if (rotation_angle < T(1e-8)) {
    rotation_aa = AngleAxisT(T(0), Vector3T(T(0), T(0), T(1)));
  }
  const Translation3T translation_tf(
          translation[0], translation[1], translation[2]);
  const Affine3T transform = translation_tf * rotation_aa;
  return transform;
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

struct LidarResidual {
    template <typename T>
    bool operator() (const T* pose_i,
                     const T* pose_j) const {
      // TODO: Every time we will match this 1 point to the best point in target
      // TODO: points, and then we will minimize that distance.
      return true;
    }

    LidarResidual(const LidarFactor& lidar_factor_i,
                  const LidarFactor& lidar_factor_j) :
                  lidar_i(lidar_factor_i.pointcloud),
                  lidar_j(lidar_factor_j.pointcloud) {}

    static AutoDiffCostFunction<LidarResidual, 3, 3>* create(
            const LidarFactor& lidar_factor_i,
            const LidarFactor& lidar_factor_j) {
      // TODO: For each point in LidarFactor, find closest correspondence
      // TODO: Maybe use iteration callback to update after every iteration?

      LidarResidual* residual = new LidarResidual(lidar_factor_i,
                                                  lidar_factor_j);
      return new AutoDiffCostFunction<LidarResidual, 3, 3>(residual);
    }

    const vector<Vector2f>& lidar_i;
    const vector<Vector2f>& lidar_j;
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
  printf("# of odometry factors: %lu\n", problem.odometry_factors.size());
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
  // TODO: Redo odometry residual for 3DOF
  // TODO: Add all odometry residuals to problem.
  // TODO: Write lidar residual.
  // TODO: Add lidar residuals against X past ones.
  // TODO: Run ceres
  // TODO: Visualize the new poses and lidar point clouds.

  // Then we will need to add the poses and the odometry as odometry residuals

  // Then we will add the last X pointclouds to the residual along with the different between their poses.
  // Which will be the to world for the beginning run.

  // Lastly we will run and have the ceres callback project the output to the world!
  ceres::Solve(options, &ceres_problem, &summary);
  printf("%s\n", summary.FullReport().c_str());
  return true;
}

