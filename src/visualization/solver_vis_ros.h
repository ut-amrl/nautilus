//
// Created by jack on 10/15/20.
//

#ifndef NAUTILUS_SOLVER_VIS_IMPL_H
#define NAUTILUS_SOLVER_VIS_IMPL_H

#include "ros/ros.h"

#include "../util/slam_types.h"
#include "./solver_vis.h"

namespace nautilus::visualization {

class SolverVisualizerROS : public SolverVisualizer {
public:
    SolverVisualizerROS(std::shared_ptr<slam_types::SLAMState2D>& state, ros::NodeHandle& n);
    void DrawSolution() const;
    void DrawCorrespondence(const Correspondence&) const;
private:
    ros::Publisher points_pub_;
    ros::Publisher poses_pub_;
};

}

#endif //NAUTILUS_SOLVER_VIS_IMPL_H
