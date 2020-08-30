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