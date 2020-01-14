//
// Created by jack on 1/3/20.
//

#ifndef LIDAR_SLAM_CORRELATIVESCANMATCHER_H
#define LIDAR_SLAM_CORRELATIVESCANMATCHER_H

#include <cstdint>
#include <memory>
#include <vector>
#include <mutex>
#include <boost/dynamic_bitset.hpp>
#include <glog/logging.h>
#include "string"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include "Eigen/Dense"

#include "./slam_type_builder.h"
#include "./slam_types.h"
#include "./timer.h"

#define DEFAULT_GAUSSIAN_SIGMA 4
#define DEFAULT_GAUSSIAN_TAP_LENGTH 11

using std::vector;
using Eigen::Vector2f;
using sensor_msgs::Image;
using slam_types::RobotPose2D;

struct LookupTable {
  uint64_t width;
  uint64_t height;
  double resolution;
  vector<vector<double>> values;
  LookupTable(const uint64_t range,
              const double resolution) :
              width((range * 2.0) / resolution + 1),
              height((range * 2.0) / resolution + 1),
              resolution(resolution) {
    values = vector<vector<double>>(height, vector<double>(width, 0));
  }

  LookupTable() : width(0), height(0), resolution(1) {}

  inline double GetPointValue(Vector2f point) const {
    uint64_t x = width / 2 + point.x() / resolution;
    uint64_t y = height / 2 + point.y() / resolution;
    if (x >= width || y >= height) {
      return 0.0;
    }
    return values[x][y];
  }

  void SetPointValue(Vector2f point, double value) {
    uint64_t x = width / 2 + point.x() / resolution;
    uint64_t y = height / 2 + point.y() / resolution;
    if (x >= width || y >= height) {
      return;
    }
    values[x][y] = value;
  }

  double GetGaussianWeight(uint64_t x, uint64_t y, const double sigma) const {
    return (1.0 / sqrt(2.0 * M_PI * sigma * sigma)) *
           pow(M_E, -static_cast<double>(x*x + y*y)/(2.0 * sigma * sigma));
  }

  double GetGaussianValue(const vector<vector<double>>& original_values,
                          uint64_t x,
                          uint64_t y,
                          const double sigma,
                          const uint64_t tap_length) const {
    double sum = 0.0;
    // When these fail we will have to re-work how we do the bounds,
    // but that would mean a massive grid size.
    CHECK_GE(x + (tap_length / 2), 0);
    CHECK_GE(y + (tap_length / 2), 0);
    for (int64_t col = x - tap_length / 2;
         col < static_cast<int64_t>(x + (tap_length / 2));
         col++) {
      for (int64_t row = y - tap_length / 2;
           row < static_cast<int64_t>(y + (tap_length / 2));
           row++) {
        if (col < 0 ||
            row < 0 ||
            col >= static_cast<int64_t>(width) ||
            row >= static_cast<int64_t>(height)) {
          continue;
        }
        sum +=
          original_values[col][row]* GetGaussianWeight(col - x, row - y, sigma);
      }
    }
    // Gaussians don't produce a total sum greater than 1,
    // but because of rounding errors it is totally possible here.
    if (sum >= 1.0) {
      sum = 1.0;
    }
    return sum;
  }

  void GaussianBlur(const double sigma, const uint64_t tap_length) {
    vector<vector<double>> result_values = values;
    std::mutex result_mtx;
    // Blur the table of values using a gaussian blur.
    #pragma omp parallel for default(none) shared(result_mtx, result_values)
    for (uint64_t col = 0; col < width; col++) {
      for (uint64_t row = 0; row < height; row++) {
        double value = GetGaussianValue(values, col, row, sigma, tap_length);
        result_mtx.lock();
        result_values[col][row] = value;
        result_mtx.unlock();
      }
    }
    values.swap(result_values);
  }

  void GaussianBlur() {
    GaussianBlur(DEFAULT_GAUSSIAN_SIGMA, DEFAULT_GAUSSIAN_TAP_LENGTH);
  }

  Image getDebugImage() const {
    Image image;
    image.header.frame_id = "map";
    image.width = width;
    image.height = height;
    image.encoding = sensor_msgs::image_encodings::MONO8;
    image.is_bigendian = 0;
    image.step = width;
    for (vector<double> row : values) {
      for (double prob : row) {
        image.data.push_back(prob * 255);
      }
    }
    return image;
  }
};

class CorrelativeScanMatcher {
 public:
    CorrelativeScanMatcher(double scanner_range, double low_res, double high_res)
    : range_(scanner_range), low_res_(low_res), high_res_(high_res) {};
    RobotPose2D GetTransformation(const vector<Vector2f>& pointcloud_a,
                                  const vector<Vector2f>& pointcloud_b);
 private:
    LookupTable GetLookupTable(const vector<Vector2f>& pointcloud, double resolution);
    LookupTable GetLookupTableHighRes(const vector<Vector2f>& pointcloud);
    LookupTable GetLookupTableLowRes(const vector<Vector2f>& pointcloud);
    std::pair<double, RobotPose2D> 
      GetProbAndTransformation(const vector<Vector2f>& pointcloud_a,
                               const vector<Vector2f>& pointcloud_b,
                               double resolution,
                               double x_min,
                               double x_max,
                               double y_min, 
                               double y_max,
                               bool excluding,
                               const boost::dynamic_bitset<>& excluded);
    double range_;
    double low_res_;
    double high_res_;
};


#endif //LIDAR_SLAM_CORRELATIVESCANMATCHER_H
