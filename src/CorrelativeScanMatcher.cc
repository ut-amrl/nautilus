//
// Created by jack on 1/3/20.
//

#include <vector>
#include <boost/dynamic_bitset.hpp>
#include "Eigen/Dense"

#include "CorrelativeScanMatcher.h"
#include "./timer.h"

using std::vector;
using Eigen::Vector2f;

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

vector<Vector2f> TranslatePointcloud(const vector<Vector2f>& pointcloud,
                                     const double x_trans,
                                     const double y_trans) {
  vector<Vector2f> translated_pointcloud;
  for (const Vector2f& point : pointcloud) {
    Vector2f translated_point(point.x() + x_trans, point.y() + y_trans);
    translated_pointcloud.push_back(translated_point);
  }
  return translated_pointcloud;
}

double CalculatePointcloudCost(const vector<Vector2f>& pointcloud,
                               const double x_trans,
                               const double y_trans,
                               const LookupTable& cost_table) {
  double probability = 0.0;
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
  // Two degree accuracy seems to be enough for now.
  for (double rotation = 0.0; rotation <= M_PI / 2; rotation += M_PI / 180) {
    // Rotate the pointcloud by this rotation.
    const vector<Vector2f> rotated_pointcloud_a =
      RotatePointcloud(pointcloud_a, rotation);
    for (double x_trans = x_min; x_trans <= x_max; x_trans += resolution) {
      for (double y_trans = y_min;
           y_trans <= y_max;
           y_trans += resolution) {
        if (excluding && excluded[((pointcloud_b_cost.height / 2) +
                                  round(y_trans / resolution)) *
                                  pointcloud_b_cost.width +
                                  ((pointcloud_b_cost.width / 2) +
                                  round(x_trans / resolution))]) {
          // Don't consider transformations that have already been found.
          continue;
        }
        double probability =
          CalculatePointcloudCost(
            rotated_pointcloud_a,
            x_trans,
            y_trans,
            pointcloud_b_cost);
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
    if (prob_and_trans_low_res.first < best_probability) {
      return best_transformation;
    }
    double x_min_high_res = prob_and_trans_low_res.second.loc.x() - low_res_;
    double x_max_high_res = prob_and_trans_low_res.second.loc.x();
    double y_min_high_res = prob_and_trans_low_res.second.loc.y() - low_res_;
    double y_max_high_res = prob_and_trans_low_res.second.loc.y();
    excluded_low_res.set(
            ((low_res_width / 2) + round(y_max_high_res / low_res_)) *
            low_res_width + ((low_res_width / 2) +
            round(x_max_high_res / low_res_)), true);
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
