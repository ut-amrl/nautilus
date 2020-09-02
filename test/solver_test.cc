//
// Created by jack on 11/24/19.
//

#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "../src/solver.h"

using Eigen::Vector2f;
using namespace nautilus;

TEST(LineSegDistanceTest, trivial_on_line) {
  Vector2f p1(0, 0);
  Vector2f p2(2, 2);
  Vector2f point_to_find(1,1);
  const LineSegment<float> line_seg(p1, p2);
  float dist = DistanceToLineSegment(point_to_find, line_seg);
  EXPECT_EQ(dist, 0) << "Point lies on the line, the distance should be 0.";
}

TEST(LineSegDistanceTest, trivial_off_line) {
  Vector2f p1(0, 0);
  Vector2f p2(2, 2);
  Vector2f point_to_find(0, 2);
  const LineSegment<float> line_seg(p1, p2);
  float dist = DistanceToLineSegment(point_to_find, line_seg);
  EXPECT_FLOAT_EQ(dist, 2.0 * sin(M_PI / 4)) << "Distance to point was incorrect";
}

TEST(LineSegDistanceTest, negative_off_line) {
  Vector2f p1(0, 0);
  Vector2f p2(2, 2);
  Vector2f point_to_find(2, 0);
  const LineSegment<float> line_seg(p1, p2);
  float dist = DistanceToLineSegment(point_to_find, line_seg);
  EXPECT_FLOAT_EQ(dist, 2.0 * sin(M_PI / 4)) << "Distance to point was incorrect";
}

TEST(LineSegDistanceTest, from_endpoint) {
  Vector2f p1(0, 0);
  Vector2f p2(2, 2);
  Vector2f point_to_find(4, 4);
  const LineSegment<float> line_seg(p1, p2);
  float dist = DistanceToLineSegment(point_to_find, line_seg);
  EXPECT_FLOAT_EQ(dist, sqrt(8)) << "Distance to point was incorrect";
}

TEST(LineSegDistanceTest, from_start) {
  Vector2f p1(0, 0);
  Vector2f p2(2, 2);
  Vector2f point_to_find(-2, -2);
  const LineSegment<float> line_seg(p1, p2);
  float dist = DistanceToLineSegment(point_to_find, line_seg);
  EXPECT_FLOAT_EQ(dist, sqrt(8)) << "Distance to point was incorrect";
}

TEST(LineSegDistanceTest, line_is_endpoint) {
  Vector2f p1(0, 0);
  Vector2f p2(2, 2);
  Vector2f point_to_find(2, 2);
  const LineSegment<float> line_seg(p1, p2);
  float dist = DistanceToLineSegment(point_to_find, line_seg);
  EXPECT_FLOAT_EQ(dist, 0) << "Distance to point was incorrect";
}
