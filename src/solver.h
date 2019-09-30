//
// Created by jack on 9/25/19.
//

#ifndef LIDAR_SLAM_SOLVER_H
#define LIDAR_SLAM_SOLVER_H

#include "ros/package.h"

#include "slam_types.h"


class solver {
  bool SolveSLAM(slam_types::SLAMProblem2D, ros::NodeHandle&);
};


#endif //LIDAR_SLAM_SOLVER_H
