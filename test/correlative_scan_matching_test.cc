//
// Created by jack on 1/4/20.
//

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../src/CorrelativeScanMatcher.h"

using Eigen::Vector2f;

TEST(LookupTableTest, GuassianBlurTest) {
  // Range of 2, and a resolution of 1 means a 5x5 table.
  LookupTable lookupTable(2, 1);
  lookupTable.SetPointValue(Vector2f(0, 0), 1);
  lookupTable.GaussianBlur(1, 11);
  ASSERT_FLOAT_EQ(0.24197072451, lookupTable.GetPointValue(Vector2f(1, 0)));
  ASSERT_FLOAT_EQ(0.24197072451, lookupTable.GetPointValue(Vector2f(0, 1)));
  ASSERT_FLOAT_EQ(0.14676266317, lookupTable.GetPointValue(Vector2f(-1, -1)));
}