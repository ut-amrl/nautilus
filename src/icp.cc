//
// Created by jack on 9/15/19.
//

#include "ros/package.h"
#include "eigen3/Eigen/Dense"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "./pointcloud_helpers.h"
#include "./gui_helpers.h"
#include "gflags/gflags.h"

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Rotation2D;
using Eigen::Matrix2d;
using sensor_msgs::PointCloud2;
using ros::Publisher;
using ceres::Solver;
using visualization_msgs::Marker;
using gui_helpers::Color4f;
using pointcloud_helpers::PublishPointcloud;

#define ERROR_SQUARED_NORM 0.1

double EuclidDistance(Vector2d p1, Vector2d p2) {
  return sqrt(pow(double(p2[0] - p1[0]), 2.0f) +
              pow(double(p2[1] - p1[1]), 2.0f));
}

Vector3d toVec3(Vector2d vec) {
  return Vector3d(vec.x(), vec.y(), 0.0);
}

std::vector<Vector2d> transform(std::vector<Vector2d> original_points, Vector2d translation, double rotation, Vector2d source_center) {
  Matrix2d rotationm = Rotation2D<double>(rotation).toRotationMatrix();
  std::vector<Vector2d> output_points;
  for (size_t index = 0; index < original_points.size(); index++) {
    output_points.push_back((rotationm * (original_points[index] - source_center)) + source_center + translation);
  }
  return output_points;
}

Vector2d findCenter(std::vector<Vector2d> points) {
  Vector2d center(0.0, 0.0);
  for (Vector2d point: points) {
    center[0] += point[0];
    center[1] += point[1];
  }
  center /= points.size();
  return center;
}

std::vector<std::pair<Vector2d, Vector2d>>
getMatches(std::vector<Vector2d> source_points,
           std::vector<Vector2d> target_points,
           Marker &match_lines) {
  std::vector<std::pair<Vector2d, Vector2d>> matches;
  for (uint64_t source_index = 0; source_index < source_points.size(); source_index++) {
    // Find the closest point
    double closest_distance = std::numeric_limits<double>::max();
    std::pair<int, int> current_match;
    for (uint64_t target_index = 0; target_index < target_points.size(); target_index++) {
      if (EuclidDistance(source_points[source_index], target_points[target_index]) < closest_distance) {
        closest_distance = EuclidDistance(source_points[source_index], target_points[target_index]);
        current_match = std::pair<int, int>(source_index, target_index);
      }
    }
    auto match = std::pair<Vector2d, Vector2d>(source_points[current_match.first], target_points[current_match.second]);
    gui_helpers::AddLine(toVec3(match.first), toVec3(match.second), Color4f::kYellow, &match_lines);
    matches.push_back(match);
  }
  return matches;
}

struct ICPError {
    std::pair<Vector2d, Vector2d> match;
    Vector2d source_center;
    ICPError(std::pair<Vector2d, Vector2d> match, Vector2d source_center)
            : match(match), source_center(source_center) {}

    template <typename T>
    bool operator()(const T* const rot,
                    const T* const x_trans,
                    const T* const y_trans,
                    T* residuals) const {
      T transformed_source_point[2];
      T angle = *rot;
      transformed_source_point[0] =
              cos(angle) * (match.first[0] - source_center[0])
              - sin(angle) * (match.first[1] - source_center[1])
              + source_center[0];
      transformed_source_point[1] =
              sin(angle) * (match.first[0] - source_center[0])
              + cos(angle) * (match.first[1] - source_center[1])
              + source_center[1];

      transformed_source_point[0] += *x_trans;
      transformed_source_point[1] += *y_trans;

      residuals[0] = (transformed_source_point[0] - match.second[0]);
      residuals[1] = (transformed_source_point[1] - match.second[1]);
      return true;
    }

    static ceres::CostFunction* Create(std::pair<Vector2d, Vector2d> match, Vector2d center) {
      return (new ceres::AutoDiffCostFunction<ICPError, 2, 1, 1, 1>(new ICPError(match, center)));
    }
};

Publisher source_publisher;
Publisher target_publisher;

std::vector<Vector2d> PerformIcp(std::vector<Vector2d> source_points,
             PointCloud2 &source_points_m,
             std::vector<Vector2d> target_points,
             PointCloud2 &target_points_m,
             bool debug) {
  ros::NodeHandle n;
  ros::Publisher line_pub = n.advertise<visualization_msgs::Marker>("/icp/match_lines", 10);
  if (debug && (source_publisher == NULL || target_publisher == NULL)) {
    source_publisher =
            n.advertise<sensor_msgs::PointCloud2>("/icp/source_points", 10);
    target_publisher =
            n.advertise<sensor_msgs::PointCloud2>("/icp/target_points", 10);
  }
  Marker match_lines;
  InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                   Color4f::kYellow,
                   0.01f,
                   0.0f,
                   0.0f,
                   &match_lines);
  Vector3d translation_rotation(0.0, 0.0, 0.0);
  for(;;) {
    Vector2d translation(0, 0);
    double rotation = 0.0f;
    if (debug) {
      PublishPointcloud(source_points, source_points_m, source_publisher);
      PublishPointcloud(target_points, target_points_m, target_publisher);
    }
    Vector2d source_center = findCenter(source_points);
    ceres::Problem problem;
    std::vector<std::pair<Vector2d, Vector2d>> matches;
    matches = getMatches(source_points, target_points, match_lines);
    for (auto match : matches) {
      problem.AddResidualBlock(ICPError::Create(match, source_center),
                               NULL,
                               &rotation,
                               &translation[0],
                               &translation[1]);
    }
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    //Update translation and rotation
    source_points = transform(source_points, translation, rotation, source_center);
    Vector3d movement(translation[0], translation[1], rotation);
    if (movement.squaredNorm() < ERROR_SQUARED_NORM) {
      return source_points;
    }
    if (debug) {
      line_pub.publish(match_lines);
      gui_helpers::ClearMarker(&match_lines);
    }
  }
}