#ifndef SOLVER_VIS_H
#define SOLVER_VIS_H

#include "../optimization/data_structures.h"
#include "../util/slam_types.h"
#include "ceres/ceres.h"

/* An abstract class used to represent what a visualizer for solver should
 * implement and how. */
namespace nautilus::visualization {

class SolverVisualizer : public ceres::IterationCallback {
 public:
  SolverVisualizer(std::shared_ptr<slam_types::SLAMState2D> state)
      : state_(state) {}

  // So that every ceres iteration callback will update the solution.
  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary&) override {
    DrawSolution();
    return ceres::SOLVER_CONTINUE;
  }

  virtual void DrawSolution() const {
    // Do nothing here.
  }

  virtual void DrawCorrespondence(const PointCorrespondences&) const {
    // Do nothing here.
  }

  virtual void DrawScans(const std::vector<int>) const {
    // Do nothing here.
  }

  virtual void DrawCovariances(std::vector<std::tuple<int, Eigen::Matrix2f>>) const {
    // Do nothing here.
  }

 protected:
  std::shared_ptr<slam_types::SLAMState2D> state_;
};

}  // namespace nautilus::visualization

#endif  // SOLVER_VIS_H
