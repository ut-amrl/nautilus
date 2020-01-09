//
// Created by jack on 1/3/20.
//

#include <vector>
#include "Eigen/Dense"

#include "CorrelativeScanMatcher.h"

using std::vector;
using Eigen::Vector2f;

LookupTable
CorrelativeScanMatcher::GetLookupTable(const vector<Vector2f>& pointcloud, double resolution) {
  LookupTable table(range_, resolution);
  for (const Vector2f& point : pointcloud) {
    table.SetPointValue(point, 1);
  }
  table.GaussianBlur();
  return table;
}

LookupTable
CorrelativeScanMatcher::GetLookupTableLowRes(const vector<Vector2f>& pointcloud) {
  return GetLookupTable(pointcloud, low_res_);
}

LookupTable
CorrelativeScanMatcher::GetLookupTableHighRes(const vector<Vector2f>& pointcloud) {
  return GetLookupTable(pointcloud, high_res_);
}

vector<Vector2f> RotatePointcloud(const vector<Vector2f>& pointcloud,
                                  const double rotation) {
  Eigen::Matrix2f rot_matrix = Eigen::Rotation2Df(rotation).toRotationMatrix();
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
                               const LookupTable& cost_table) {
  double probability = 1.0;
  for (const Vector2f& point : pointcloud) {
    probability *= cost_table.GetPointValue(point);
  }
  return probability;
}

RobotPose2D
CorrelativeScanMatcher::GetTransformation(const vector<Vector2f>& pointcloud_a,
                                          const vector<Vector2f>& pointcloud_b) {
  const LookupTable pointcloud_b_cost = GetLookupTable(pointcloud_b, high_res_);
  RobotPose2D current_most_likely_trans;
  double current_most_likely_prob = 0.0;
  std::mutex most_likely_mtx;
  // Two degree accuracy seems to be enough for now.
  for (double rotation = 0.0; rotation <= M_PI / 2; rotation += M_PI / 180) {
    // Rotate the pointcloud by this rotation.
    const vector<Vector2f> rotated_pointcloud_a =
      RotatePointcloud(pointcloud_a, rotation);
    for (double x_trans = -range_; x_trans <= range_; x_trans += high_res_) {
      for (double y_trans = -range_;
           y_trans <= range_;
           y_trans += high_res_) {
        double probability =
          CalculatePointcloudCost(
            TranslatePointcloud(rotated_pointcloud_a, x_trans, y_trans),
            pointcloud_b_cost);
        most_likely_mtx.lock();
        if (probability > current_most_likely_prob) {
          current_most_likely_trans = RobotPose2D(Vector2f(x_trans, y_trans),
                                                  rotation);
        }
        most_likely_mtx.unlock();
      }
    }
    std::cout << "Increasing rotation" << std::endl;
  }
  return current_most_likely_trans;
}
