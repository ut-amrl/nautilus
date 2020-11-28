//
// Created by jack on 11/14/20.
//

#include <vector>
#include <memory>

#include "Eigen/Dense"

#include "../util/slam_util.h"
#include "../util/slam_types.h"
#include "./lc_candidate_filter.h"

namespace nautilus::loop_closure {

using std::shared_ptr;
using slam_types::SLAMState2D;
using std::vector;
using Eigen::Vector2f;

LCCandidateFilter::LCCandidateFilter(shared_ptr<SLAMState2D> state) : state_(state) {};

    Eigen::Vector2f ComputeMean(const vector<Vector2f> &pointcloud) {
      // Compute the mean and return the mean vector.
      Eigen::Vector2f mean_vector(0, 0);
      for (const auto &p : pointcloud) {
        mean_vector += p;
      }
      return (1.0 / pointcloud.size()) * mean_vector;
    }

/// @desc: Computes the min eigenvalue / max eigenvalue of a scatter matrix for
/// a particular scan.
/// @param pointcloud: The pointcloud to compute the scatter matrix of.
double ComputeScatterMatrixScore(const vector<Vector2f> &pointcloud) {
  Vector2f mean = ComputeMean(pointcloud);
  Eigen::Matrix2f scatter_matrix;
  scatter_matrix << 0, 0, 0, 0;
  // Compute the scatter matrix.
  for (const auto &p : pointcloud) {
    scatter_matrix += (p - mean) * (p - mean).transpose();
  }
  Eigen::EigenSolver<Eigen::Matrix2f> eigen_solver;
  eigen_solver.compute(scatter_matrix);
  // Now extract the eigen values.
  auto eigen_values = eigen_solver.eigenvalues();
  CHECK_EQ(eigen_values.rows(), 2);
  double ev_1 = eigen_values(0, 0).real();
  double ev_2 = eigen_values(1, 0).real();
  return std::min(ev_1, ev_2) / std::max(ev_1, ev_2);
}

bool DistantFromLastScan(std::shared_ptr<slam_types::SLAMState2D> state,
                         size_t node_idx, std::vector<size_t> scans,
                         double distance) {
  if (scans.empty()) {
    return true;
  }
  auto last_scan_trans = GetPoseTranslation(state, scans[scans.size() - 1]);
  auto node_trans = GetPoseTranslation(state, node_idx);
  return (node_trans - last_scan_trans).norm() >= distance;
}

std::vector<size_t> LCCandidateFilter::GetLCCandidates() {
  vector<size_t> scans;
  for (size_t i = 0; i < state_->problem.nodes.size(); i++) {
    if (!DistantFromLastScan(state_, i, scans, 5)) {
      // Skip close scans.
      continue;
    }
    double score = ComputeScatterMatrixScore(
            state_->problem.nodes[i].lidar_factor.pointcloud);
    // A score close to 1 means a good spread in both axes of the scan.
    // So a high score means a good location for loop closure because it has
    // good spread.
    if (score >= 0.70) {
      scans.push_back(i);
    }
  }
  return scans;
};


}  // nautilus::loop_closure