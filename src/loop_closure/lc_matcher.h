//
// Created by jack on 11/14/20.
//

#ifndef NAUTILUS_LC_MATCHER_H
#define NAUTILUS_LC_MATCHER_H

#include <vector>
#include <memory>

#include "ceres/ceres.h"

#include "../util/slam_types.h"

namespace nautilus::loop_closure {

class LCMatcher {
public:
  /// @desc: This class matches given scans with other scans based on the uncertainty pulled out of the last
  ///        optimization.
  /// @param state: The current state of the problem including most likely poses.
  /// @param problem: The problem used in the initial optimization.
  LCMatcher(std::shared_ptr<slam_types::SLAMState2D> state,
            std::shared_ptr<ceres::Problem> problem);
  /// @desc: Finds all the probable matches to the source scan contained within the candidates list given.
  /// @param source: The index of the scan to find all candidates that could match to it.
  /// @param match_candidates: The list of candidates in which matches will be drawn from.
  /// @returns: A vector of indices for the scans that are possible matches to source.
  std::vector<size_t> GetPossibleMatches(size_t source, std::vector<size_t> match_candidates);
  /// @desc: Gets the "score" between two scans. This is a measure of the likelihood that they match based on the
  ///        uncertainty from the initial solution.
  /// @param source: The first scan to set to constant and evaluate the uncertainty of target from.
  /// @param target: The second scan whose uncertainty will be measured to see if source is close enough.
  /// @returns: A tuple with the first element being the covariance and the second element is the "score".
  std::tuple<Eigen::Matrix2f, double> ChiSquareScore(size_t source, size_t target);
private:
  const std::shared_ptr<slam_types::SLAMState2D> state_;
  const std::shared_ptr<ceres::Problem> problem_;
};

}

#endif //NAUTILUS_LC_MATCHER_H
