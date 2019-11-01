//
// Created by jack on 9/25/19.
//

#ifndef SRC_SOLVER_H_
#define SRC_SOLVER_H_

#include <vector>

#include <ros/node_handle.h>
#include "ros/package.h"
#include "eigen3/Eigen/Dense"
#include "ceres/ceres.h"

#include "./kdtree.h"
#include "./slam_types.h"

using std::vector;
using Eigen::Vector2f;

class Solver {
 public:
  struct PointCorrespondences {
    vector<Vector2f> source_points;
    vector<Vector2f> target_points;
    vector<Vector2f> target_normals;
    double *source_pose;
    double *target_pose;
    PointCorrespondences(double* source_pose, double* target_pose)
    : source_pose(source_pose), target_pose(target_pose) {}
  };
  Solver(double translation_weight,
         double rotation_weight,
         double stopping_accuracy);
  bool SolveSLAM(slam_types::SLAMProblem2D&, ros::NodeHandle&);
  double GetPointCorrespondences(const slam_types::SLAMProblem2D& problem,
                                 vector<slam_types::SLAMNodeSolution2D>*
                                   solution_ptr,
                                 PointCorrespondences* point_correspondences,
                                 size_t source_node_index,
                                 size_t target_node_index);
  void AddOdomFactors(const slam_types::SLAMProblem2D& problem,
                      vector<slam_types::SLAMNodeSolution2D>& solution,
                      ceres::Problem* ceres_problem);

private:
  double translation_weight;
  double rotation_weight;
  double stopping_accuracy;
};

#endif // SRC_SOLVER_H_
