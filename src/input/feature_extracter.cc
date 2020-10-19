//
// Created by jack on 9/13/20.
//

#include <vector>
#include <Eigen/Dense>
#include "glog/logging.h"
#include <iostream>

#include "feature_extracter.h"

namespace nautilus {

std::vector<Eigen::Vector2f> GetNeighborhood(const std::vector<Eigen::Vector2f>& points,
                                             size_t point_index,
                                             size_t neighbors_per_side,
                                             double max_neighbor_distance) {
  std::vector<Eigen::Vector2f> neighbors;
  // Left side neighbors.
  for (size_t neighbor_idx = std::max(static_cast<size_t>(0), point_index - neighbors_per_side);
       neighbor_idx < point_index; neighbor_idx++) {
    if ((points[point_index] - points[neighbor_idx]).norm() <= max_neighbor_distance) {
      neighbors.push_back(points[neighbor_idx]);
    }
  }
  // Right side neighbors.
  for (size_t neighbor_idx = point_index + 1; neighbor_idx < std::min(points.size(), point_index + neighbors_per_side); neighbor_idx++) {
    neighbors.push_back(points[neighbor_idx]);
  }
  return neighbors;
}

Eigen::Vector2f ComputeMean(const std::vector<Eigen::Vector2f>& neighborhood) {
  // Compute the mean and return the mean vector.
  Eigen::Vector2f mean_vector(0,0);
  for (const auto& p : neighborhood) {
    mean_vector += p;
  }
  return (1.0 / neighborhood.size()) * mean_vector;
}

std::vector<std::pair<double, Eigen::Vector2f>> ComputeSmoothnessScores(const std::vector<Eigen::Vector2f>& points,
                                                                        std::vector<float>* unsorted_scores,
                                                                        int neighbors_per_side,
                                                                        double max_neighbor_distance,
                                                                        int min_neighbor_num) {
  // For each point used the smoothness formula.
  // This formula is computing a scatter matrix for every point and its neighborhood.
  // Then the smoothness score is the smallest eigenvalue / largest eigenvalue.
  std::vector<std::pair<double, Eigen::Vector2f>> smoothness_scores;
  for (size_t i = 0; i < points.size(); i++) {
    const auto& point = points[i];
    // Get the points around point, and then include point.
    std::vector<Eigen::Vector2f> neighborhood = GetNeighborhood(points, i, neighbors_per_side, max_neighbor_distance);
    if (neighborhood.size() < static_cast<size_t>(min_neighbor_num)) {
      // Skip this iteration if not enough neighbors.
      continue;
    }
    neighborhood.push_back(point);
    // Compute the neighborhood of point and all the points around it.
    const Eigen::Vector2f mean_vector = ComputeMean(neighborhood);
    // Now get the scatter matrix using the formula summation from 1 to n of all points
    // (Xi - m)(Xi - m)^T
    Eigen::Matrix2f scatter_matrix;
    scatter_matrix << 0,0,0,0;
    for (const auto& p : neighborhood) {
      Eigen::Matrix2f temp = (p - mean_vector) * ((p - mean_vector).transpose());
      scatter_matrix += temp;
    }
    Eigen::EigenSolver<Eigen::Matrix2f> eigen_solver;
    eigen_solver.compute(scatter_matrix);
    // Smoothness score is smaller eigen value / larger eigen value so it is [0, 1].
    auto eigen_values = eigen_solver.eigenvalues();
    CHECK_EQ(eigen_values.rows(), 2);
    double eigen_value_1 = eigen_values(0, 0).real();
    double eigen_value_2 = eigen_values(1, 0).real();
    double smoothness_score = std::min(eigen_value_1, eigen_value_2) / std::max(eigen_value_1, eigen_value_2);
    if (smoothness_score < 0 || smoothness_score > 1) {
      std::cout << smoothness_score << std::endl;
      std::cout << "Eigen Values: " << eigen_value_1 << " " << eigen_value_2 << std::endl;
    }
    smoothness_scores.emplace_back(smoothness_score, point);
    unsorted_scores->push_back(smoothness_score);
   }
  return smoothness_scores;
}

FeatureExtractor::FeatureExtractor(const std::vector<Eigen::Vector2f>& points,
                                   double threshold,
                                   double distance_threshold,
                                   int neighbor_num,
                                   int max_edge_num,
                                   int max_planar_num,
                                   int min_neighbor_num)
                                   : threshold_(threshold),
                                     distance_threshold_(distance_threshold),
                                     neighbors_per_side_(neighbor_num),
                                     max_edge_number_(max_edge_num),
                                     max_planar_number_(max_planar_num),
                                     min_neighbor_num_(min_neighbor_num) {
  // Compute the smoothness scores of every point once.
  smoothness_points_ = ComputeSmoothnessScores(points, &unordered_scores_, neighbors_per_side_, max_neighbor_distance_, min_neighbor_num_);
  // Now sort the points based on their smoothness.
  std::sort(smoothness_points_.begin(), smoothness_points_.end(),
           [](const std::pair<double, Eigen::Vector2f>& point_a,
              const std::pair<double, Eigen::Vector2f>& point_b) {
    return point_a.first < point_b.first;
  });
}

bool validFeaturePoint(std::pair<double, Eigen::Vector2f> point,
                       std::vector<Eigen::Vector2f> points,
                       double threshold,
                       double distance_threshold,
                       size_t max_size,
                       bool is_edge=false) {
  // Planar points must be less than the threshold, and edge points must be more than the threshold.
  if (!is_edge && point.first > threshold) {
    return false;
  }
  if (is_edge && point.first < threshold) {
    return false;
  }
  // We can accept points.
  if (points.size() >= max_size) {
    return false;
  }
  // Not close to any of the other points.
  for (const auto& p : points) {
    if ((p - point.second).norm() < distance_threshold) {
      return false;
    }
  }
  return true;
}

std::vector<Eigen::Vector2f> FeatureExtractor::GetPlanarPoints() {
  std::vector<Eigen::Vector2f> planar_points;
  // Super simple O(n^2) algorithm to find the planar points.
  for (size_t i = 0; i < smoothness_points_.size(); i++) {
    if (validFeaturePoint(smoothness_points_[i], planar_points, threshold_, distance_threshold_, max_planar_number_)) {
      planar_points.push_back(smoothness_points_[i].second);
    }
  }
  return planar_points;
}

std::vector<Eigen::Vector2f> FeatureExtractor::GetEdgePoints() {
  std::vector<Eigen::Vector2f> edge_points;
  for (int i = smoothness_points_.size() - 1; i >= 0; i--) {
    if (validFeaturePoint(smoothness_points_[i], edge_points, threshold_, distance_threshold_, max_edge_number_, true)) {
      edge_points.push_back(smoothness_points_[i].second);
    }
  }
  return edge_points;
}

std::vector<float> FeatureExtractor::GetSmoothnessScores() {
  return unordered_scores_;
}
}
