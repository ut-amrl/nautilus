//
// Created by jack on 11/14/20.
//

#include <vector>
#include <thread>
#include <memory>

#include "ceres/ceres.h"

#include "../util/slam_util.h"
#include "../util/slam_types.h"
#include "./lc_matcher.h"

namespace nautilus::loop_closure {

using slam_types::SLAMState2D;
using std::vector;
using std::shared_ptr;
using ceres::Problem;
using std::tuple;

LCMatcher::LCMatcher(shared_ptr<SLAMState2D> state, shared_ptr<Problem> problem) :
        state_(state), problem_(problem) {};


// Gets the Covariance matrix between any two scans.
Eigen::Matrix2f GetCovarianceMatrix(shared_ptr<SLAMState2D> state, Problem *problem, size_t source, size_t target) {
  ceres::Covariance::Options options;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  ceres::Covariance cov(options);
  vector<std::pair<const double *, const double *>> blocks;
  blocks.emplace_back(state->solution[source].pose,
                      state->solution[target].pose);
  double values[9] = {0};
  problem->SetParameterBlockVariable(state->solution[0].pose);
  // TODO: Check more than 0
  problem->SetParameterBlockConstant(state->solution[std::min(source, target) - 1].pose);
  CHECK(cov.Compute(blocks, problem));
  CHECK(cov.GetCovarianceBlock(state->solution[source].pose, state->solution[target].pose, values));
  problem->SetParameterBlockVariable(state->solution[std::min(source, target) - 1].pose);
  problem->SetParameterBlockConstant(state->solution[0].pose);
  Eigen::Matrix2d covariance_mat;  //= Eigen::Map<Eigen::Matrix3d>(values);
  covariance_mat << values[0], values[1], values[3], values[4];
  return covariance_mat.cast<float>();
}

// Gets the score between the source and the target node. A representation of
// how likely they are to match for loop closure.
tuple<Eigen::Matrix2f, double> LCMatcher::ChiSquareScore(size_t source, size_t target) {
  auto covariance_mat = GetCovarianceMatrix(state_, problem_.get(), source, target);
  Eigen::Vector2f source_pose = GetPoseTranslation(state_, source);
  Eigen::Vector2f target_pose = GetPoseTranslation(state_, target);
  double score = (target_pose - source_pose).transpose() * covariance_mat.inverse() *
                 (target_pose - source_pose);
  return {covariance_mat, score};
}

vector<size_t> LCMatcher::GetPossibleMatches(size_t source, vector<size_t> match_candidates) {
  CHECK_GT(match_candidates.size(), 0);
  vector<size_t> matches;
  for (size_t target_scan : match_candidates) {
    if (target_scan == source) {
      // A scan can't match itself.
      continue;
    }
    auto [cov, score] = ChiSquareScore(source, target_scan);
    // If score below threshold then it is a good match as it is probable within the uncertainty.
    if (score < 5000.0) {
      matches.push_back(target_scan);
    }
  }
  return matches;
}
}
