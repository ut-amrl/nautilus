//
// Created by jack on 11/24/19.
//

#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"

using Eigen::Vector2f;

TEST(LineSegDistanceTest, Trvial) {
  Vector2f p1(0, 0);
  Vector2f p2(2, 2);
  Vector2f point_to_find(1,1);
  EXPECT_EQ(p1, p2)
}