//
// Created by jack on 1/4/20.
//

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../src/CImg.h"
#include "../src/slam_types.h"
#include "../src/CorrelativeScanMatcher.h"

using Eigen::Vector2f;
using slam_types::RobotPose2D;

TEST(CorrelativeScanMatcherTest, GetTransformationBasicTest) {
  CorrelativeScanMatcher scan_matcher(4, 0.3, 0.03);
  vector<Vector2f> pointcloud_a;
  for (double i = -3; i < 3; i += 0.03) {
    // Draw two lines in the scan, like a perfect hallway.
    pointcloud_a.push_back(Vector2f(i, 1));
    pointcloud_a.push_back(Vector2f(i, -1));
  }
  vector<Vector2f> pointcloud_b;
  // Rotate by 45 degrees and shift by 10 and 10, try and see if we can't get
  // the same exact results back.
  Eigen::Matrix2f a_to_b =
    Eigen::Rotation2Df(M_PI / 4).toRotationMatrix();
  for (const Vector2f& point : pointcloud_a) {
    pointcloud_b.push_back(a_to_b * point);
  }
  RobotPose2D pose = scan_matcher.GetTransformation(pointcloud_a, pointcloud_b);
  ASSERT_LE(fabs(pose.angle - M_PI / 4), 0.00001);
}
