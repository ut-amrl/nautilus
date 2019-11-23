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
using slam_types::SLAMNode2D;
using lidar_slam::HitlSlamInputMsgConstPtr;

struct LCPoses {
    vector<int> a_poses_;
    vector<int> b_poses_;
    Eigen::Hyperplane<float, 2> line_a_;
    Eigen::Hyperplane<float, 2> line_b_;
    LCPoses(vector<int>& a_poses,
            vector<int>& b_poses,
            Eigen::Hyperplane<float, 2> line_a,
            Eigen::Hyperplane<float, 2> line_b) :
            a_poses_(a_poses),
            b_poses_(b_poses),
            line_a_(line_a),
            line_b_(line_b) {}
};

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
         double lc_translation_weight,
         double lc_rotation_weight,
         double stopping_accuracy,
         SLAMProblem2D& problem,
         ros::NodeHandle& n);
  vector<SLAMNodeSolution2D> SolveSLAM();
  double GetPointCorrespondences(const SLAMProblem2D& problem,
                                 vector<SLAMNodeSolution2D>*
                                   solution_ptr,
                                 PointCorrespondences* point_correspondences,
                                 size_t source_node_index,
                                 size_t target_node_index);
  void AddOdomFactors(const vector<OdometryFactor2D>& odom_factors,
                      vector<SLAMNodeSolution2D>& solution,
                      ceres::Problem* ceres_problem,
                      double trans_weight,
                      double rot_weight);
  void HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr);
  vector<SLAMNodeSolution2D> GetSolution() {
    return solution_;
  }
  void AddColinearConstraints(Eigen::Hyperplane<float, 2> line_a,
                              Eigen::Hyperplane<float, 2> line_b,
                              vector<vector<SLAMNode2D>> poses);
  void SolveForLC();
  void AddColinearResiduals(ceres::Problem* problem);
  double AddLidarResidualsForLC(ceres::Problem& problem);
private:
  double translation_weight_;
  double rotation_weight_;
  double lc_translation_weight_;
  double lc_rotation_weight_;
  double stopping_accuracy_;
  SLAMProblem2D problem_;
  vector<SLAMNodeSolution2D> solution_;
  ros::NodeHandle n_;
  vector<LCPoses> loop_closure_constraints_;
};

#endif // SRC_SOLVER_H_
