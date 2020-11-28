//
// Created by jack on 11/14/20.
//

#ifndef NAUTILUS_LC_CANDIDATE_FILTER_H
#define NAUTILUS_LC_CANDIDATE_FILTER_H

#include <vector>
#include <memory>

#include "../util/slam_types.h"

namespace nautilus::loop_closure {

class LCCandidateFilter {
public:
  /// @desc: Not every scan is good for loop closure. This class will filter out the bad scans and return
  ///        only those that seem promising for being unique.
  /// @param state: The shared state to the currently being solved SLAM Problem.
  LCCandidateFilter(std::shared_ptr<slam_types::SLAMState2D> state);
  /// @desc: Returns scans that are good for loop closure. This is classified as having a good spread
  ///        in both the x and y axises. A good candidate will also be far from other candidates.
  /// @returns: A vector of indices for each of the scans that should be looked at for Loop Closure.
  std::vector<size_t>  GetLCCandidates();
private:
  const std::shared_ptr<slam_types::SLAMState2D> state_;
};

}

#endif //NAUTILUS_LC_CANDIDATE_FILTER_H
