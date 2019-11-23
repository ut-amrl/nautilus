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
#include "lidar_slam/HitlSlamInputMsg.h"

using std::vector;
using Eigen::Vector2f;
using slam_types::OdometryFactor2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using lidar_slam::HitlSlamInputMsgConstPtr;

class Solver {
 public:
  struct PointCorrespondences {
    vector<Vector2f> source_points;
    vector<Vector2f> target_points;
    vector<Vector2f> source_normals;
    vector<Vector2f> target_normals;
    double *source_pose;
    double *target_pose;
    PointCorrespondences(double* source_pose, double* target_pose)
    : source_pose(source_pose), target_pose(target_pose) {}
  };
  Solver(double translation_weight,
         double rotation_weight,
         double stopping_accuracy,
         SLAMProblem2D& problem);
  vector<SLAMNodeSolution2D> SolveSLAM(ros::NodeHandle&);
  double GetPointCorrespondences(const SLAMProblem2D& problem,
                                 vector<SLAMNodeSolution2D>*
                                   solution_ptr,
                                 PointCorrespondences* point_correspondences,
                                 size_t source_node_index,
                                 size_t target_node_index);
  void AddOdomFactors(const vector<OdometryFactor2D>& odom_factors,
                      vector<SLAMNodeSolution2D>& solution,
                      ceres::Problem* ceres_problem);
  void HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr);
  vector<SLAMNodeSolution2D> GetSolution() {
    return solution_;
  }
private:
  double translation_weight_;
  double rotation_weight_;
  double stopping_accuracy_;
  SLAMProblem2D problem_;
  vector<SLAMNodeSolution2D> solution_;
};

#endif // SRC_SOLVER_H_
