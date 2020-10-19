//
// Created by jack on 10/15/20.
//

#ifndef NAUTILUS_SLAM_UTIL_H
#define NAUTILUS_SLAM_UTIL_H

#include <vector>

#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "../optimization/data_structures.h"

namespace nautilus {

/* Implements the common SLAM utility functions for transforming by poses and the like */

template<typename T>
inline Eigen::Transform<T, 2, Eigen::Affine> PoseArrayToAffine(
        const T *rotation, const T *translation) {
  typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
  typedef Eigen::Rotation2D<T> Rotation2DT;
  typedef Eigen::Translation<T, 2> Translation2T;
  Affine2T affine = Translation2T(translation[0], translation[1]) *
                    Rotation2DT(rotation[0]).toRotationMatrix();
  return affine;
}

template<typename T>
inline Eigen::Transform<T, 2, Eigen::Affine> PoseArrayToAffine(
        const T *pose_array) {
  return PoseArrayToAffine(&pose_array[2], &pose_array[0]);
}

inline std::vector<Eigen::Vector2f> TransformPointcloud(double *pose,
                                                        const std::vector<Eigen::Vector2f>& pointcloud) {
  std::vector<Eigen::Vector2f> pcloud;
  Eigen::Affine2f trans = PoseArrayToAffine(&pose[2], &pose[0]).cast<float>();
  for (const Eigen::Vector2f &p : pointcloud) {
    pcloud.push_back(trans * p);
  }
  return pcloud;
}

// Reference from:
// https://github.com/SoylentGraham/libmv/blob/master/src/libmv/simple_pipeline/bundle.cc
inline Eigen::MatrixXd CRSToEigen(const ceres::CRSMatrix &crs_matrix) {
  Eigen::MatrixXd matrix(crs_matrix.num_rows, crs_matrix.num_cols);
  matrix.setZero();
  // Row contains starting position of this row in the cols.
  for (int row = 0; row < crs_matrix.num_rows; row++) {
    int row_start = crs_matrix.rows[row];
    int row_end = crs_matrix.rows[row + 1];
    // Cols contains the non-zero elements column numbers.
    for (int col = row_start; col < row_end; col++) {
      int col_num = crs_matrix.cols[col];
      // Value is contained in the same index of the values array.
      double value = crs_matrix.values[col];
      matrix(row, col_num) = value;
    }
  }
  return matrix;
}

// Returns if val is between a and b.
template<typename T>
bool IsBetween(const T &val, const T &a, const T &b) {
  return (val >= a && val <= b) || (val >= b && val <= a);
}

template<typename T>
T DistanceToLineSegment(const Eigen::Matrix<T, 2, 1> &point,
                        const LineSegment<T> &line_seg) {
  typedef Eigen::Matrix<T, 2, 1> Vector2T;
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

}

#endif //NAUTILUS_SLAM_UTIL_H
