#pragma once

#include "eigen3/Eigen/Dense"

template <typename T>
Eigen::Transform<T, 2, Eigen::Affine> PoseArrayToAffine(const T* rotation,
                                                        const T* translation) {
  using Rotation2DT = Eigen::Rotation2D<T>;
  using Translation2T = Eigen::Translation<T, 2>;
  return Translation2T(translation[0], translation[1]) *
         Rotation2DT(rotation[0]).toRotationMatrix();
}

template <typename T>
Eigen::Transform<T, 2, Eigen::Affine> PoseArrayToAffine(const T* pose_array) {
  return PoseArrayToAffine(&pose_array[2], &pose_array[0]);
}

// Returns if val is between a and b.
template <typename T>
bool IsBetween(const T& val, const T& a, const T& b) {
  return (val >= a && val <= b) || (val >= b && val <= a);
}

template <typename T>
T DistanceToLineSegment(const Eigen::Matrix<T, 2, 1>& point,
                        const nautilus::ds::LineSegment<T>& line_seg) {
  using Vector2T = Eigen::Matrix<T, 2, 1>;
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