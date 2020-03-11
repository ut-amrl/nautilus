//
// Created by jack on 2/21/20.
//

#ifndef CURATINGLONGTERMVECTORMAPS_LINE_EXTRACTION_H
#define CURATINGLONGTERMVECTORMAPS_LINE_EXTRACTION_H

#include <vector>

#include "glog/logging.h"
#include "Eigen/Dense"

#define SAME_POINT_EPSILON 0.000001

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

        double Parameterize(const Vector2f& point) const {
          typedef Eigen::Hyperplane<float, 2> Line2D;
          const Line2D infinite_line = Line2D::Through(start_point, end_point);
          Vector2f point_projection = infinite_line.projection(point);
          Vector2f diff_vec = end_point - start_point;
          if (abs(diff_vec.x()) > SAME_POINT_EPSILON) {
            return (point_projection.x() - start_point.x()) / diff_vec.x();
          } else if (abs(diff_vec.y()) > SAME_POINT_EPSILON) {
            return (point_projection.y() - start_point.y()) / diff_vec.y();
          } else {
            // The start and endpoints are so close that we should treat them as one point.
            // Return -1 so that the distance calculation will just return the distance
            // to one of the endpoints, either one, doesn't matter.
            return -1;
          }
        }

        double DistanceToLineSegment(const Vector2f& point) const {
          // Project the point onto the line.
          typedef Eigen::Hyperplane<float, 2> Line2D;
          // Parameterize according to the projection.
          double t = Parameterize(point);
          if (t < 0) {
            // This point is closest to the start point.
            return (start_point - point).norm();
          } else if (t > 1) {
            // This point is closest to the end point.
            return (point - end_point).norm();
          } else {
            // Otherwise, the point is closest to the line.
            // This is equivalent to the orthogonal projection.
            const Line2D infinite_line = Line2D::Through(start_point, end_point);
            const Vector2f point_projection = infinite_line.projection(point);
            return (point - point_projection).norm();
          }
        }

        bool PointOnLine(const Vector2f& point, double threshold) const {
          // Parameterize the point.
          double t = Parameterize(point);
          if (t < 0 || t > 1) {
            // This point is closest to the start point.
            // Or to the end point
            return false;
          }
          double dist_to_line = DistanceToLineSegment(point);
          return dist_to_line < threshold;
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
