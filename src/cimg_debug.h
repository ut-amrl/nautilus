//
// Created by jack on 2/28/20.
//

#ifndef ROS_DEBUG_TOOLS_CIMG_DEBUG_H
#define ROS_DEBUG_TOOLS_CIMG_DEBUG_H

#include <vector>

#include "CImg.h"

using cimg_library::CImg;
using cimg_library::CImgDisplay;

struct WrappedImage {
  const uint64_t width;
  const uint64_t height;
  double resolution;
  CImg<double> values;
  WrappedImage(const double range, const double resolution)
      : width(floor((range * 2.0) / resolution)),
        height(floor((range * 2.0) / resolution)),
        resolution(resolution) {
    // Construct a width x height image, with only 1 z level.
    // And, only one double per color with default value 0.0.
    values = CImg<double>(width, height, 1, 1, 0.0);
  }

  WrappedImage() : width(0), height(0), resolution(1) {}

  inline uint64_t convertX(float x) const {
    return width / 2 + floor(x / resolution);
  }

  inline uint64_t convertY(float y) const {
    return height / 2 + floor(y / resolution);
  }

  inline double GetPointValue(Vector2f point) const {
    uint64_t x = convertX(point.x());
    uint64_t y = convertY(point.y());
    return values(x, y);
  }

  void SetPointValue(Vector2f point, double value) {
    uint64_t x = convertX(point.x());
    uint64_t y = convertY(point.y());
    if (x >= width || y >= height) {
      return;
    }
    values(x, y) = value;
  }

  CImg<double> GetDebugImage() const { return values; }
};

inline WrappedImage GetTable(const vector<Vector2f>& pointcloud, double range,
                             double resolution) {
  WrappedImage table(range, resolution);
  for (const Vector2f& point : pointcloud) {
    table.SetPointValue(point, 1);
  }
  return table;
}

inline Vector2f furthest_point(const vector<Vector2f>& points) {
  Vector2f point;
  double dist = 0;
  for (const Vector2f& p : points) {
    if (p.norm() > dist) {
      dist = p.norm();
      point = p;
    }
  }
  return point;
}

inline WrappedImage DrawPoints(const vector<Vector2f>& points) {
  double width = furthest_point(points).norm();
  // Plot the points in a display.
  WrappedImage table = GetTable(points, width, 0.03);
  return table;
}

inline WrappedImage DrawLine(const Vector2f& start_point,
                             const Vector2f& end_point, WrappedImage image) {
  double color[] = {1.0};
  image.values.draw_line(
      image.convertX(start_point.x()), image.convertY(start_point.y()),
      image.convertX(end_point.x()), image.convertY(end_point.y()), color);
  return image;
}

inline void WaitForClose(WrappedImage image) {
  cimg_library::CImgDisplay display(image.GetDebugImage());
  while (!display.is_closed()) {
    display.wait();
  }
}

inline void WaitForClose(std::vector<WrappedImage> images) {
  std::vector<cimg_library::CImgDisplay> displays;
  for (auto image : images) {
    displays.emplace_back(image.GetDebugImage());
    displays.at(displays.size() - 1).show();
  }
  while (!displays[0].is_closed()) {
    displays[0].wait();
  }
}

inline void SaveImage(std::string name, WrappedImage image) {
  image.GetDebugImage().normalize(0, 255).save_bmp(name.c_str());
}

#endif  // ROS_DEBUG_TOOLS_CIMG_DEBUG_H
