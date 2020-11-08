#ifndef SLAM_RESIDUALS_H
#define SLAM_RESIDUALS_H

#include <vector>

#include "../util/slam_types.h"
#include "./data_structures.h"
#include "Eigen/Geometry"
#include "ceres/ceres.h"

/* This file contains all the residuals (error functions) used in the
 * optimization process to solve for the maximum likelihood map.
 */

namespace nautilus {

struct OdometryResidual {
  template <typename T>
  bool operator()(const T *pose_i, const T *pose_j, T *residual) const {
    // Predicted pose_j = pose_i * odometry.
    // Hence, error = pose_j.inverse() * pose_i * odometry;
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
    // Extract the translation.
    const Vector2T Ti(pose_i[0], pose_i[1]);
    const Vector2T Tj(pose_j[0], pose_j[1]);
    // The Error in the translation is the difference with the odometry
    // in the direction of the previous pose, then getting rid of the new
    // rotation (transpose = inverse for rotation matrices).
    const Vector2T error_translation = Ti + T_odom.cast<T>() - Tj;
    // Rotation error is very similar to the translation error, except
    // we don't care about the difference in the position.
    const T rotation_diff = pose_i[2] + T(R_odom) - pose_j[2];
    const T error_rotation = atan2(sin(rotation_diff), cos(rotation_diff));
    // The residuals are weighted according to the parameters set
    // by the user.
    residual[0] = T(translation_weight) * error_translation.x();
    residual[1] = T(translation_weight) * error_translation.y();
    residual[2] = T(rotation_weight) * error_rotation;
    return true;
  }

  OdometryResidual(const slam_types::OdometryFactor2D &factor,
                   double translation_weight, double rotation_weight)
      : translation_weight(translation_weight),
        rotation_weight(rotation_weight),
        R_odom(factor.rotation),
        T_odom(factor.translation) {}

  static ceres::AutoDiffCostFunction<OdometryResidual, 3, 3, 3> *create(
      const slam_types::OdometryFactor2D &factor, double translation_weight,
      double rotation_weight) {
    OdometryResidual *residual =
        new OdometryResidual(factor, translation_weight, rotation_weight);
    return new ceres::AutoDiffCostFunction<OdometryResidual, 3, 3, 3>(residual);
  }

  double translation_weight;
  double rotation_weight;
  const float R_odom;
  const Eigen::Vector2f T_odom;
};

// Lidar Normal Residual
struct LIDARNormalResidual {
  template <typename T>
  bool operator()(const T *source_pose, const T *target_pose,
                  T *residuals) const {
    typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
    const Affine2T source_to_world =
        PoseArrayToAffine(&source_pose[2], &source_pose[0]);
    const Affine2T world_to_target =
        PoseArrayToAffine(&target_pose[2], &target_pose[0]).inverse();
    const Affine2T source_to_target = world_to_target * source_to_world;
#pragma omp parallel for shared(residuals)
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

  LIDARNormalResidual(const std::vector<Eigen::Vector2f> &source_points,
                      const std::vector<Eigen::Vector2f> &target_points,
                      const std::vector<Eigen::Vector2f> &source_normals,
                      const std::vector<Eigen::Vector2f> &target_normals)
      : source_points(source_points),
        target_points(target_points),
        source_normals(source_normals),
        target_normals(target_normals) {
    CHECK_EQ(source_points.size(), target_points.size());
    CHECK_EQ(target_points.size(), target_normals.size());
    CHECK_EQ(source_normals.size(), target_normals.size());
  }

  static ceres::AutoDiffCostFunction<LIDARNormalResidual, ceres::DYNAMIC, 3, 3>
      *create(const std::vector<Eigen::Vector2f> &source_points,
              const std::vector<Eigen::Vector2f> &target_points,
              const std::vector<Eigen::Vector2f> &source_normals,
              const std::vector<Eigen::Vector2f> &target_normals) {
    CHECK_GT(source_points.size(), 0);
    LIDARNormalResidual *residual = new LIDARNormalResidual(
        source_points, target_points, source_normals, target_normals);
    return new ceres::AutoDiffCostFunction<LIDARNormalResidual, ceres::DYNAMIC,
                                           3, 3>(residual,
                                                 source_points.size() * 2);
  }

  const std::vector<Eigen::Vector2f> source_points;
  const std::vector<Eigen::Vector2f> target_points;
  const std::vector<Eigen::Vector2f> source_normals;
  const std::vector<Eigen::Vector2f> target_normals;
};

struct LIDARPointResidual {
  template <typename T>
  bool operator()(const T *source_pose, const T *target_pose,
                  T *residuals) const {
    typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
    const Affine2T source_to_world =
        PoseArrayToAffine(&source_pose[2], &source_pose[0]);
    const Affine2T world_to_target =
        PoseArrayToAffine(&target_pose[2], &target_pose[0]).inverse();
    const Affine2T source_to_target = world_to_target * source_to_world;
#pragma omp parallel for shared(residuals)
    for (size_t index = 0; index < source_points.size(); index++) {
      Vector2T source_pointT = source_points[index].cast<T>();
      Vector2T target_pointT = target_points[index].cast<T>();
      // Transform source_point into the frame of target_point
      source_pointT = source_to_target * source_pointT;
      Vector2T difference_in_target = target_pointT - source_pointT;
      residuals[index * 2] = difference_in_target(0);
      residuals[index * 2 + 1] = difference_in_target(1);
    }
    return true;
  }

  LIDARPointResidual(const std::vector<Eigen::Vector2f> &source_points,
                     const std::vector<Eigen::Vector2f> &target_points,
                     const std::vector<Eigen::Vector2f> &source_normals,
                     const std::vector<Eigen::Vector2f> &target_normals)
      : source_points(source_points),
        target_points(target_points),
        source_normals(source_normals),
        target_normals(target_normals) {
    CHECK_EQ(source_points.size(), target_points.size());
    CHECK_EQ(target_points.size(), target_normals.size());
    CHECK_EQ(source_normals.size(), target_normals.size());
  }

  static ceres::AutoDiffCostFunction<LIDARPointResidual, ceres::DYNAMIC, 3, 3>
      *create(const std::vector<Eigen::Vector2f> &source_points,
              const std::vector<Eigen::Vector2f> &target_points,
              const std::vector<Eigen::Vector2f> &source_normals,
              const std::vector<Eigen::Vector2f> &target_normals) {
    CHECK_GT(source_points.size(), 0);
    LIDARPointResidual *residual = new LIDARPointResidual(
        source_points, target_points, source_normals, target_normals);
    return new ceres::AutoDiffCostFunction<LIDARPointResidual, ceres::DYNAMIC,
                                           3, 3>(residual,
                                                 source_points.size() * 2);
  }

  const std::vector<Eigen::Vector2f> source_points;
  const std::vector<Eigen::Vector2f> target_points;
  const std::vector<Eigen::Vector2f> source_normals;
  const std::vector<Eigen::Vector2f> target_normals;
};

struct PointToLineResidual {
  template <typename T>
  bool operator()(const T *pose, const T *line_pose, T *residuals) const {
    typedef Eigen::Matrix<T, 2, 1> Vector2T;
    typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
    const Affine2T pose_to_world = PoseArrayToAffine(&pose[2], &pose[0]);
    const Affine2T line_to_world =
        PoseArrayToAffine(&line_pose[2], &line_pose[0]);
    Vector2T line_start = line_to_world * line_segment_.start.cast<T>();
    Vector2T line_end = line_to_world * line_segment_.end.cast<T>();
    const LineSegment<T> TransformedLineSegment(line_start, line_end);
#pragma omp parallel for shared(residuals)
    for (size_t index = 0; index < points_.size(); index++) {
      Vector2T pointT = points_[index].cast<T>();
      // Transform source_point into the frame of the line
      pointT = pose_to_world * pointT;
      T dist_along_normal =
          DistanceToLineSegment(pointT, TransformedLineSegment);
      residuals[index] = dist_along_normal;
    }
    return true;
  }

  PointToLineResidual(const LineSegment<float> &line_segment,
                      const vector<Vector2f> points)
      : line_segment_(line_segment), points_(points) {}

  static ceres::AutoDiffCostFunction<PointToLineResidual, ceres::DYNAMIC, 3, 3>
      *create(const LineSegment<float> &line_segment,
              const vector<Vector2f> points) {
    PointToLineResidual *res = new PointToLineResidual(line_segment, points);
    return new ceres::AutoDiffCostFunction<PointToLineResidual, ceres::DYNAMIC,
                                           3, 3>(res, points.size());
  }

  const LineSegment<float> line_segment_;
  const std::vector<Eigen::Vector2f> points_;
};

}  // namespace nautilus

#endif
