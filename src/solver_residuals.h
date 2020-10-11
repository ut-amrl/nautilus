#pragma once

#include "eigen3/Eigen/Dense"

#include "slam_types.h"
#include "slam_utils.h"
#include "solver_datastructures.h"

namespace nautilus {
namespace residuals {

/*----------------------------------------------------------------------------*
 *                             CERES RESIDUALS                                |
 *----------------------------------------------------------------------------*/

struct OdometryResidual {
  template <typename T>
  bool operator()(const T* pose_i, const T* pose_j, T* residual) const {
    // Predicted pose_j = pose_i * odometry.
    // Hence, error = pose_j.inverse() * pose_i * odometry;
    using Vector2T = Eigen::Matrix<T, 2, 1>;
    // Extract the translation.
    const Vector2T Ti(pose_i[0], pose_i[1]);
    const Vector2T Tj(pose_j[0], pose_j[1]);
    // The Error in the translation is the difference with the odometry
    // in the direction of the previous pose, then getting rid of the new
    // rotation (transpose = inverse for rotation matrices).
    const Vector2T error_translation = Ti + T_odom.cast<T>() - Tj;
    // Rotation error is very similar to the translation error, except
    // we don't care about the difference in the position.
    const T error_rotation = pose_i[2] + T(R_odom) - pose_j[2];
    // The residuals are weighted according to the parameters set
    // by the user.
    residual[0] = T(translation_weight) * error_translation.x();
    residual[1] = T(translation_weight) * error_translation.y();
    residual[2] = T(rotation_weight) * error_rotation;
    return true;
  }

  OdometryResidual(const slam_types::OdometryFactor2D& factor,
                   double translation_weight,
                   double rotation_weight)
      : translation_weight(translation_weight),
        rotation_weight(rotation_weight),
        R_odom(factor.rotation),
        T_odom(factor.translation) {}

  static ceres::AutoDiffCostFunction<OdometryResidual, 3, 3, 3>* create(
      const slam_types::OdometryFactor2D& factor,
      double translation_weight,
      double rotation_weight) {
    return new ceres::AutoDiffCostFunction<OdometryResidual, 3, 3, 3>(
        new OdometryResidual(factor, translation_weight, rotation_weight));
  }

  double translation_weight;
  double rotation_weight;
  const float R_odom;
  const Eigen::Vector2f T_odom;
};

struct LIDARPointBlobResidual {
  // TODO: Add the source normals penalization as well.
  // Would cause there to be two normals.
  template <typename T>
  bool operator()(const T* source_pose,
                  const T* target_pose,
                  T* residuals) const {
    using Affine2T = Eigen::Transform<T, 2, Eigen::Affine>;
    using Vector2T = Eigen::Matrix<T, 2, 1>;
    const Affine2T source_to_world =
        PoseArrayToAffine(&source_pose[2], &source_pose[0]);
    const Affine2T world_to_target =
        PoseArrayToAffine(&target_pose[2], &target_pose[0]).inverse();
    const Affine2T source_to_target = world_to_target * source_to_world;
#pragma omp parallel for default(none) shared(residuals)
    for (size_t index = 0; index < source_points.size(); index++) {
      Vector2T source_pointT = source_points[index].cast<T>();
      Vector2T target_pointT = target_points[index].cast<T>();
      // Transform source_point into the frame of target_point
      source_pointT = source_to_target * source_pointT;
      T target_normal_result =
          target_normals[index].cast<T>().dot(source_pointT - target_pointT);
      T source_normal_result =
          source_normals[index].cast<T>().dot(target_pointT - source_pointT);
      residuals[index * 2] = target_normal_result;
      residuals[index * 2 + 1] = source_normal_result;
    }
    return true;
  }

  LIDARPointBlobResidual(std::vector<Eigen::Vector2f>& source_points,
                         std::vector<Eigen::Vector2f>& target_points,
                         std::vector<Eigen::Vector2f>& source_normals,
                         std::vector<Eigen::Vector2f>& target_normals)
      : source_points(source_points),
        target_points(target_points),
        source_normals(source_normals),
        target_normals(target_normals) {
    CHECK_EQ(source_points.size(), target_points.size());
    CHECK_EQ(target_points.size(), target_normals.size());
    CHECK_EQ(source_normals.size(), target_normals.size());
  }

  static ceres::
      AutoDiffCostFunction<LIDARPointBlobResidual, ceres::DYNAMIC, 3, 3>*
      create(std::vector<Eigen::Vector2f>& source_points,
             std::vector<Eigen::Vector2f>& target_points,
             std::vector<Eigen::Vector2f>& source_normals,
             std::vector<Eigen::Vector2f>& target_normals) {
    auto residual = new LIDARPointBlobResidual(
        source_points, target_points, source_normals, target_normals);
    return new ceres::
        AutoDiffCostFunction<LIDARPointBlobResidual, ceres::DYNAMIC, 3, 3>(
            residual, source_points.size() * 2);
  }

  const std::vector<Eigen::Vector2f> source_points;
  const std::vector<Eigen::Vector2f> target_points;
  const std::vector<Eigen::Vector2f> source_normals;
  const std::vector<Eigen::Vector2f> target_normals;
};

struct PointToLineResidual {
  template <typename T>
  bool operator()(const T* pose, const T* line_pose, T* residuals) const {
    using Vector2T = Eigen::Matrix<T, 2, 1>;
    using Affine2T = Eigen::Transform<T, 2, Eigen::Affine>;
    const Affine2T pose_to_world = PoseArrayToAffine(pose);
    const Affine2T line_to_world = PoseArrayToAffine(line_pose);
    Vector2T line_start = line_to_world * line_segment_.start.cast<T>();
    Vector2T line_end = line_to_world * line_segment_.end.cast<T>();
    const ds::LineSegment<T> world_line_segment(line_start, line_end);
#pragma omp parallel for default(none) shared(residuals)
    for (size_t i = 0; i < points_.size(); i++) {
      // Transform source_point into the frame of the line
      Vector2T world_point = pose_to_world * points_[i].cast<T>();
      residuals[i] = DistanceToLineSegment(world_point, world_line_segment);
    }
    return true;
  }

  PointToLineResidual(const ds::LineSegment<float>& line_segment,
                      const std::vector<Eigen::Vector2f>& points)
      : line_segment_(line_segment), points_(points) {}

  static ceres::AutoDiffCostFunction<PointToLineResidual, ceres::DYNAMIC, 3, 3>*
  create(const ds::LineSegment<float>& line_segment,
         const std::vector<Eigen::Vector2f>& points) {
    return new ceres::
        AutoDiffCostFunction<PointToLineResidual, ceres::DYNAMIC, 3, 3>(
            new PointToLineResidual(line_segment, points), points.size());
  }

  const ds::LineSegment<float> line_segment_;
  const std::vector<Eigen::Vector2f> points_;
};

}  // namespace residuals
}  // namespace nautilus