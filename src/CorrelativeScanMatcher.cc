//
// Created by jack on 1/3/20.
//

#include <vector>

#include <boost/dynamic_bitset.hpp>
#include "Eigen/Dense"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"

#include "CorrelativeScanMatcher.h"
#include "./timer.h"
#include "./solver.h"
#include "./CImg.h"
#include "./slam_types.h"
#include "./pointcloud_helpers.h"
#include "./math_util.h"

using std::vector;
using Eigen::Vector2f;

using slam_types::SLAMNode2D;
using slam_types::RobotPose2D;
using slam_types::SLAMProblem2D;
using sensor_msgs::PointCloud2;

#define COST_GAMMA 1.0 / 50.0

void
CorrelativeScanMatcher::TestCorrelativeScanMatcher(ros::NodeHandle& n,
                                                   SLAMProblem2D problem,
                                                   vector<SLAMNodeSolution2D> solution,
                                                   const size_t pose_a,
                                                   const size_t pose_b) {
  PointCloud2 pointCloud_a;
  PointCloud2 pointCloud_b;
  pointcloud_helpers::InitPointcloud(&pointCloud_a);
  pointcloud_helpers::InitPointcloud(&pointCloud_b);
  ros::Publisher pub_a = n.advertise<PointCloud2>("/corr_pose_a", 10);
  ros::Publisher pub_b = n.advertise<PointCloud2>("/corr_pose_b", 10);
  std::cout << "Comparing node " << pose_a << " with node " << pose_b << std::endl;
  std::cout << "Publishing pointclouds" << std::endl;
  vector<Vector2f> point_a;
  point_a.push_back(Vector2f(solution[pose_a].pose[0], solution[pose_a].pose[1]));
  vector<Vector2f> point_b;
  point_b.push_back(Vector2f(solution[pose_b].pose[0], solution[pose_b].pose[1]));
  sleep(1);
  for (int i = 0; i < 5; i++) {
    pointcloud_helpers::PublishPointcloud(point_a, pointCloud_a, pub_a);
    pointcloud_helpers::PublishPointcloud(point_b, pointCloud_b, pub_b);
    ros::spinOnce();
  }
  CHECK_LT(pose_a, problem.nodes.size());
  CHECK_LT(pose_b, problem.nodes.size());
  SLAMNode2D& node_a = problem.nodes[pose_a];
  SLAMNode2D& node_b = problem.nodes[pose_b];
  auto pointcloud_a = node_a.lidar_factor.pointcloud;
  auto pointcloud_b = node_b.lidar_factor.pointcloud;
  Vector2f loc_a(solution[pose_a].pose[0], solution[pose_a].pose[1]);
  Vector2f loc_b(solution[pose_b].pose[0], solution[pose_b].pose[1]);
  std::cout << "Testing Scan Get Transformation" << std::endl;
  CorrelativeScanMatcher scan_matcher(2, 0.3, 0.03);
  RobotPose2D transformation = scan_matcher.GetTransformation(pointcloud_a, pointcloud_b);
  std::cout << "Done" << std::endl;
  std::cout << "Found transformation: " << std::endl << "Translation: " << transformation.loc << std::endl << "Rotation: " << math_util::RadToDeg(transformation.angle) << std::endl;
  Eigen::Affine2f trans = Eigen::Translation2f(transformation.loc) * Eigen::Rotation2Df(transformation.angle);
  vector<Vector2f> new_pointcloud_b;
  for (const Vector2f& point : pointcloud_b) {
    new_pointcloud_b.emplace_back(trans * point);
  }
  LookupTable a_lookup = scan_matcher.GetLookupTableHighRes(pointcloud_a);
  LookupTable b_lookup = scan_matcher.GetLookupTableHighRes(new_pointcloud_b);
  LookupTable old_b_lookup = scan_matcher.GetLookupTableHighRes(pointcloud_b);
  LookupTable old_b_low_res_lookup = scan_matcher.GetLookupTableLowRes(old_b_lookup);
  cimg_library::CImgDisplay display;
  cimg_library::CImgDisplay old_display_b;
  cimg_library::CImgDisplay old_b_low_res;
  cimg_library::CImg<double> a_img = a_lookup.GetDebugImage();
  cimg_library::CImg<double> b_img = b_lookup.GetDebugImage();
  old_display_b.display(old_b_lookup.GetDebugImage());
  old_b_low_res.display(old_b_low_res_lookup.GetDebugImage());
  cimg_library::CImg<double> mix_img(a_img.width(), a_img.height(), 1, 3, 0);
  for (int x = 0; x < a_img.width(); x++) {
    for (int y = 0; y < a_img.height(); y++) {
      mix_img(x, y, 0, 0) = a_img(x, y);
      mix_img(x, y, 0, 1) = b_img(x, y);
    }
  }
  display.display(mix_img);
  std::cout << "Displaying debug images" << std::endl;
  while (!display.is_closed()) {}
}

LookupTable
CorrelativeScanMatcher::GetLookupTable(const vector<Vector2f>& pointcloud,
                                       double resolution) {
  LookupTable table(range_, resolution);
  for (const Vector2f& point : pointcloud) {
    table.SetPointValue(point, 1);
  }
  table.GaussianBlur();
  return table;
}

LookupTable
CorrelativeScanMatcher::GetLookupTableLowRes(const LookupTable& high_res_table) {
  LookupTable low_res_table(range_, low_res_);
  // Run the max filter over the portions of this table.
  for (double x = -range_; x <= range_; x += low_res_) {
    for (double y = -range_; y <= range_; y += low_res_) {
      // Get the max value for all the cells that this low res
      // cell encompasses.
      double max = high_res_table.MaxArea(x - low_res_, y - low_res_, x, y);
      low_res_table.SetPointValue(Vector2f(x, y), max);
    }
  }
  return low_res_table;
}

LookupTable
CorrelativeScanMatcher::GetLookupTableHighRes(const vector<Vector2f>& pointcloud) {
  return GetLookupTable(pointcloud, high_res_);
}

vector<Vector2f> RotatePointcloud(const vector<Vector2f>& pointcloud,
                                  const double rotation) {
  const Eigen::Matrix2f rot_matrix =
    Eigen::Rotation2Df(rotation).toRotationMatrix();
  vector<Vector2f> rotated_pointcloud;
  for (const Vector2f& point : pointcloud) {
    rotated_pointcloud.push_back(rot_matrix * point);
  }
  return rotated_pointcloud;
}

double CalculatePointcloudCost(const vector<Vector2f>& pointcloud,
                               const double x_trans,
                               const double y_trans,
                               const LookupTable& cost_table) {
  double probability = 1.0;
  for (const Vector2f& point : pointcloud) {
    probability += cost_table.GetPointValue(point + Vector2f(x_trans, y_trans));
  }
  probability /= pointcloud.size();
  return probability;
}

std::pair<double, RobotPose2D>
CorrelativeScanMatcher::GetProbAndTransformation(const vector<Vector2f>& pointcloud_a,
                                                 const LookupTable& pointcloud_b_cost,
                                                 double resolution,
                                                 double x_min,
                                                 double x_max,
                                                 double y_min, 
                                                 double y_max,
                                                 bool excluding,
                                                 const boost::dynamic_bitset<>& excluded) {
  RobotPose2D current_most_likely_trans;
  double current_most_likely_prob = 0.0;
  // One degree accuracy seems to be enough for now.
  for (double rotation = 0.0; rotation <= M_2PI; rotation += M_PI / 180) {
    // Rotate the pointcloud by this rotation.
    const vector<Vector2f> rotated_pointcloud_a =
      RotatePointcloud(pointcloud_a, rotation);
    for (double x_trans = x_min + resolution; x_trans <= x_max; x_trans += resolution) {
      for (double y_trans = y_min + resolution;
           y_trans <= y_max;
           y_trans += resolution) {
        // If we are excluding scans, and this is a banned scan. Then don't
        // consider it.
        if (excluding &&
            excluded[pointcloud_b_cost.AbsCoords(x_trans, y_trans)]) {
          continue;
        }
        // Otherwise, get the probability / cost of this scan.
        double probability =
          CalculatePointcloudCost(
            rotated_pointcloud_a,
            x_trans,
            y_trans,
            pointcloud_b_cost);
        // If it is the best so far, keep track of it!
        if (probability > current_most_likely_prob) {
          current_most_likely_trans = RobotPose2D(Vector2f(x_trans, y_trans),
                                                  rotation);
          current_most_likely_prob = probability;
        }
      }
    }
  }
  return std::pair<double, RobotPose2D>(current_most_likely_prob,
                                        current_most_likely_trans);
}

RobotPose2D
CorrelativeScanMatcher::GetTransformation(const vector<Vector2f>& pointcloud_a,
                                          const vector<Vector2f>& pointcloud_b) {
  double current_probability = 1.0;
  double best_probability = 0.0;
  RobotPose2D best_transformation;
  uint64_t low_res_width = (range_ * 2.0) / low_res_ + 1;
  boost::dynamic_bitset<> excluded_low_res(low_res_width * low_res_width);
  boost::dynamic_bitset<> excluded_high_res(0); // Dumby value, never used.
  const LookupTable pointcloud_b_cost_high_res = GetLookupTableHighRes(pointcloud_b);
  const LookupTable pointcloud_b_cost_low_res =
    GetLookupTableLowRes(pointcloud_b_cost_high_res);
  while (current_probability >= best_probability) {
    // Evaluate over the low_res lookup table.
    std::cout << "Low Res Iteration" << std::endl;
    auto prob_and_trans_low_res =
      GetProbAndTransformation(pointcloud_a,
                               pointcloud_b_cost_low_res,
                               low_res_,
                               -range_,
                               range_,
                               -range_,
                               range_,
                               true,
                               excluded_low_res);
    std::cout << "Low Res Transformation Best: " << std::endl;
    std::cout << prob_and_trans_low_res.second.loc.x() << " " << prob_and_trans_low_res.second.loc.y() << std::endl;
    std::cout << prob_and_trans_low_res.second.angle << std::endl;
    if (prob_and_trans_low_res.first < best_probability) {
      return best_transformation;
    }
    double x_min_high_res = prob_and_trans_low_res.second.loc.x() - low_res_;
    double x_max_high_res = prob_and_trans_low_res.second.loc.x();
    double y_min_high_res = prob_and_trans_low_res.second.loc.y() - low_res_;
    double y_max_high_res = prob_and_trans_low_res.second.loc.y();
    CHECK_GE(x_min_high_res, -range_);
    CHECK_LT(x_min_high_res, range_);
    CHECK_GE(y_min_high_res, -range_);
    CHECK_LT(y_min_high_res, range_);
    CHECK_LE(x_max_high_res, range_);
    CHECK_LE(y_max_high_res, range_);
    excluded_low_res.set(pointcloud_b_cost_low_res.AbsCoords(x_max_high_res,
                                                             y_max_high_res),
                         true);
    std::cout << "High res iteration" << std::endl;
    auto prob_and_trans_high_res = GetProbAndTransformation(pointcloud_a,
                                                            pointcloud_b_cost_high_res,
                                                            high_res_,
                                                            x_min_high_res,
                                                            x_max_high_res,
                                                            y_min_high_res,
                                                            y_max_high_res,
                                                            false,
                                                            excluded_high_res);
    if (prob_and_trans_high_res.first > best_probability) {
      // This is the new best and we should keep searching to make
      // sure there is nothing better.
      best_probability = prob_and_trans_high_res.first;
      best_transformation = prob_and_trans_high_res.second;
    }
    current_probability = prob_and_trans_high_res.first;
  }
  return best_transformation;
}
