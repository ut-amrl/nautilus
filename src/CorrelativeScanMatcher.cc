//
// Created by jack on 1/3/20.
//

#include <vector>
#include "Eigen/Dense"

#include "CorrelativeScanMatcher.h"

using std::vector;
using Eigen::Vector2f;

LookupTable
CorrelativeScanMatcher::GetLookupTable(const vector<Vector2f> pointcloud,
                                       const double range,
                                       const double resolution) {
  LookupTable table(range, resolution);
  for (const Vector2f& point : pointcloud) {
    table.SetPointValue(point, 1);
  }
  //table.GaussianBlur();
  return table;
}
