//
// Created by jack on 9/25/19.
//

#include "solver.h"
#include "slam_types.h"

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
      return true;
    }

    OdometryResidual(const OdometryFactor& factor) :
            R_odom(factor.rotation), T_odom(factor.translation) {}

    static AutoDiffCostFunction<OdometryResidual,6, 6, 6>* create(
            const OdometryFactor& factor) {
      OdometryResidual* residual = new OdometryResidual(factor);
      return new AutoDiffCostFunction<OdometryResidual, 6, 6, 6>(residual);
    }

    const Matrix3f R_odom;
    const Vector3f T_odom;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

bool solver::SolveSLAM(slam_types::SLAMProblem2D& problem) {
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
  return true;
}