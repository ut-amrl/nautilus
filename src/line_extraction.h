//
// Created by jack on 2/21/20.
//

#ifndef CURATINGLONGTERMVECTORMAPS_LINE_EXTRACTION_H
#define CURATINGLONGTERMVECTORMAPS_LINE_EXTRACTION_H

#include <vector>

#include "glog/logging.h"
#include "Eigen/Dense"

using std::vector;
using Eigen::Vector2f;

namespace VectorMaps {

    struct LineSegment {
        Vector2f start_point;
        Vector2f end_point;

        LineSegment(Vector2f start_point, Vector2f end_point) :
                start_point(start_point),
                end_point(end_point) {}

        LineSegment() {}

        double DistanceToLineSegment(Vector2f point) const {
          // Project the point onto the line.
          typedef Eigen::Hyperplane<float, 2> Line2D;
          const Line2D infinite_line = Line2D::Through(start_point, end_point);
          const Vector2f point_projection = infinite_line.projection(point);
          // Parameterize according to the projection.
          Vector2f diff_vec = end_point - start_point;
          // TODO: Make sure x diff isn't zero.
          double t =
                  (point_projection.x() - start_point.x()) / diff_vec.x();
          //CHECK_DOUBLE_EQ((point_projection.y() - start_point.y()) / diff_vec.y(), t);
          if (t < 0) {
            // This point is closest to the start point.
            return (start_point - point).norm();
          } else if (t > 1) {
            // This point is closest to the end point.
            return (point - end_point).norm();
          } else {
            // Otherwise, the point is closest to the line.
            // This is equivalent to the orthogonal projection.
            return (point - point_projection).norm();
          }
        }

        bool PointOnLine(Vector2f point, double threshold) const {
          // Project the point onto the line.
          typedef Eigen::Hyperplane<float, 2> Line2D;
          const Line2D infinite_line = Line2D::Through(start_point, end_point);
          const Vector2f point_projection = infinite_line.projection(point);
          // Parameterize according to the projection.
          double t =
                  (point_projection - start_point).norm() /
                  (end_point - start_point).norm();
          if (t < 0 || t > 1) {
            // This point is closest to the start point.
            // Or to the end point
            return false;
          }
          double dist_to_line = DistanceToLineSegment(point);
          if (dist_to_line > threshold) {
            return false;
          }
          return true;
        }
    };

    struct LineCovariances {
        const Eigen::Matrix2f start_point_cov;
        const Eigen::Matrix2f end_point_cov;

        LineCovariances() : start_point_cov(Eigen::Matrix2f::Identity()), end_point_cov(Eigen::Matrix2f::Identity()) {}

        LineCovariances(const Eigen::Matrix2f& start_point_cov, const Eigen::Matrix2f& end_point_cov) :
                start_point_cov(start_point_cov), end_point_cov(end_point_cov) {}
    };

    struct SensorCovParams {
        // Standard Deviation for Range and Angle.
        double std_dev_range = 0;
        double std_dev_angle = 0;
        SensorCovParams(double std_dev_range, double std_dev_angle) :
                std_dev_range(std_dev_range), std_dev_angle(std_dev_angle) {}
    };

    vector<LineSegment> ExtractLines(const vector<Vector2f>& pointcloud);
    Eigen::Matrix2f GetSensorCovariance(const vector<double> ranges, const vector<double> angles);
    vector<LineCovariances> GetLineEndpointCovariances(const vector<LineSegment> lines, const vector<Vector2f>& points);

}

#endif //CURATINGLONGTERMVECTORMAPS_LINE_EXTRACTION_H
