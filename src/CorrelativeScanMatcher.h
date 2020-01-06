//
// Created by jack on 1/3/20.
//

#ifndef LIDAR_SLAM_CORRELATIVESCANMATCHER_H
#define LIDAR_SLAM_CORRELATIVESCANMATCHER_H

#include <cstdint>
#include <memory>
#include <vector>
#include <glog/logging.h>
#include "string.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"

#include "Eigen/Dense"

using std::vector;
using Eigen::Vector2f;
using sensor_msgs::Image;

#define GAUSSIAN_BLUR_COEFFICIENT 1
#define GAUSSIAN_TAP_SIDE_SIZE 11

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

    double GetPointValue(Vector2f point) {
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

    double GetGaussianWeight(uint64_t x, uint64_t y) {
      const double sigma = GAUSSIAN_BLUR_COEFFICIENT;
      return (1.0 / sqrt(2.0 * M_PI * sigma * sigma)) *
             pow(M_E, -static_cast<double>(x*x + y*y)/(2.0 * sigma * sigma));
    }

    void ApplyGaussianKernel(const vector<vector<double>>& original_values,
                             vector<vector<double>>& result_values,
                             uint64_t x,
                             uint64_t y) {
      double sum = 0.0;
      // When these fail we will have to re-work how we do the bounds,
      // but that would mean a massive grid size.
      CHECK_GE(x + (GAUSSIAN_TAP_SIDE_SIZE / 2), 0);
      CHECK_GE(y + (GAUSSIAN_TAP_SIDE_SIZE / 2), 0);
      for (int64_t col = x - GAUSSIAN_TAP_SIDE_SIZE / 2; col <
              static_cast<int64_t>(x + (GAUSSIAN_TAP_SIDE_SIZE / 2)); col++) {
        for (int64_t row = y - GAUSSIAN_TAP_SIDE_SIZE / 2; row <
                static_cast<int64_t>(y + (GAUSSIAN_TAP_SIDE_SIZE / 2)); row++) {
          if (col < 0 || row < 0 || col >= static_cast<int64_t>(width) || row >= static_cast<int64_t>(height)) {
            continue;
          }
          sum +=
            original_values[col][row]* GetGaussianWeight(col - x, row - y);
        }
      }
      // No probability can be higher than 1. No gaussian will have a sum higher than
      // 1 anyways, but to be safe.
      CHECK_LE(sum, 1.0);
      result_values[x][y] = sum;
    }

    void GaussianBlur() {
      vector<vector<double>> result_values = values;
      // Blur the table of values using a gaussian blur.
      for (uint64_t col = 0; col < width; col++) {
        for (uint64_t row = 0; row < height; row++) {
          ApplyGaussianKernel(values, result_values, col, row);
        }
      }
      values.swap(result_values);
    }

    Image getDebugImage() {
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
    LookupTable GetLookupTable(const vector<Vector2f> pointcloud,
                               const double range,
                               const double resolution);
};


#endif //LIDAR_SLAM_CORRELATIVESCANMATCHER_H
