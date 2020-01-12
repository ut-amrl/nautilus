//
// Created by jack on 1/4/20.
//

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../src/slam_types.h"
#include "../src/CorrelativeScanMatcher.h"

using Eigen::Vector2f;
using slam_types::RobotPose2D;

TEST(LookupTableTest, GuassianBlurTest) {
  // Range of 2, and a resolution of 1 means a 5x5 table.
  LookupTable lookupTable(2, 1);
  lookupTable.SetPointValue(Vector2f(0, 0), 1);
  lookupTable.GaussianBlur(1, 11);
  ASSERT_FLOAT_EQ(0.24197072451, lookupTable.GetPointValue(Vector2f(1, 0)));
  ASSERT_FLOAT_EQ(0.24197072451, lookupTable.GetPointValue(Vector2f(0, 1)));
  ASSERT_FLOAT_EQ(0.14676266317, lookupTable.GetPointValue(Vector2f(-1, -1)));
}

TEST(CorrelativeScanMatcherTest, GetTransformationBasicTest) {
  CorrelativeScanMatcher scan_matcher(30, 0.3, 0.03);
  vector<Vector2f> pointcloud_a;
  for (double i = -10; i < 10; i += 0.03) {
    // Draw two lines in the scan, like a perfect hallway.
    pointcloud_a.push_back(Vector2f(i, 1));
    pointcloud_a.push_back(Vector2f(i, -1));
    std::cout << pointcloud_a.size() << " " << i << std::endl;
  }
  vector<Vector2f> pointcloud_b;
  // Rotate by 45 degrees and shift by 10 and 10, try and see if we can't get
  // the same exact results back.
  Eigen::Affine2f a_to_b =
    Eigen::Translation2f(10, 10) * Eigen::Rotation2Df(M_PI / 4);
  for (const Vector2f& point : pointcloud_a) {
    pointcloud_b.push_back(a_to_b * point);
  }
  RobotPose2D pose = scan_matcher.GetTransformation(pointcloud_a, pointcloud_b);
  ASSERT_LE(fabs(pose.angle - M_PI / 4), 0.00001);
  ASSERT_LE(fabs(pose.loc.x() -  10.0), 0.1);
  ASSERT_LE(fabs(pose.loc.y() - 10.0), 0.1);
}
