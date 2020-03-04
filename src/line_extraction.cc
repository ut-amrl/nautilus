/*
 * Work Based on "Curating Long-Term Vector Maps" by Samer Nashed and Joydeep Biswas.
 */

#include <time.h>
#include <vector>
#include <thread>

#include "Eigen/Dense"
#include "ceres/ceres.h"

#include "line_extraction.h"
#include <DEBUG.h>

using std::vector;
using std::pair;
using std::make_pair;
using std::sort;
using Eigen::Vector2f;
using ceres::AutoDiffCostFunction;

#define INLIER_THRESHOLD 0.10
#define CONVERGANCE_THRESHOLD 0.01
#define NEIGHBORHOOD_SIZE 0.50
#define NEIGHBORHOOD_GROWTH_SIZE 0.15
#define UNCERTAINTY_SAMPLE_NUM 1000

namespace VectorMaps {

// Returns a list of inliers to the line_segment
// with their indices attached as the first member of the pair.
vector<pair<int, Vector2f>> GetInliers(const LineSegment line,
                                       const vector<Vector2f> pointcloud) {
  vector<pair<int, Vector2f>> inliers;
  for (size_t i = 0; i < pointcloud.size(); i++) {
    if (line.DistanceToLineSegment(pointcloud[i]) < INLIER_THRESHOLD) {
      inliers.push_back(make_pair(i, pointcloud[i]));
    }
  }
  return inliers;
}

LineSegment RANSACLineSegment(const vector<Vector2f> pointcloud) {
  CHECK_GT(pointcloud.size(), 1);
  size_t max_possible_pairs = pointcloud.size() * (pointcloud.size() - 1);
  LineSegment best_segment;
  size_t max_inlier_num = 0;
  srand(time(NULL));
  // Loop through all pairs, or 1000 hundred of them if it's more than that
  for (size_t i = 0;
       i < std::min(max_possible_pairs, static_cast<size_t>(1000));
       i++) {
    // Find all inliers between two different random points.
    size_t start_point_index = rand() % pointcloud.size();
    Vector2f start_point = pointcloud[start_point_index];
    Vector2f end_point;
    size_t end_point_index;
    do {
      end_point_index = rand() % pointcloud.size();
      end_point = pointcloud[end_point_index];
    } while (end_point_index == start_point_index);
    // Calculate the number of inliers, if more than best pair so far, save it.
    LineSegment line(start_point, end_point);
    size_t inlier_num = GetInliers(line, pointcloud).size();
    if (inlier_num > max_inlier_num) {
      max_inlier_num = inlier_num;
      best_segment = line;
    }
  }
  CHECK_GT(max_inlier_num, 0);
  return best_segment;
}

vector<Vector2f> GetNeighborhood(const vector<Vector2f> points) {
  // Pick a random point to center this around.
  vector<Vector2f> remaining_points = points;
  srand(time(NULL));
  vector<Vector2f> neighborhood;
  do {
    neighborhood.clear();
    if (remaining_points.size() <= 0) {
      return vector<Vector2f>();
    }
    size_t rand_idx = rand() % remaining_points.size();
    Vector2f center_point = remaining_points[rand_idx];
    // Now get the points around it
    for (Vector2f point : remaining_points) {
      if ((center_point - point).norm() <= NEIGHBORHOOD_SIZE) {
        neighborhood.push_back(point);
      }
    }
    if (neighborhood.size() <= 1) {
      remaining_points.erase(remaining_points.begin() + rand_idx);
    }
  } while(neighborhood.size() <= 1);
  CHECK_GT(neighborhood.size(), 1);
  return neighborhood;
}

vector<Vector2f> GetNeighborhoodAroundLine(const LineSegment& line, const vector<Vector2f> points) {
  vector<Vector2f> neighborhood;
  for (const Vector2f& p : points) {
    if (line.DistanceToLineSegment(p) <= NEIGHBORHOOD_GROWTH_SIZE) {
      neighborhood.push_back(p);
    }
  }
  return neighborhood;
}

Vector2f GetCenterOfMass(const vector<Vector2f>& pointcloud) {
  Vector2f sum(0,0);
  for (const Vector2f& p : pointcloud) {
    sum += p;
  }
  return sum / pointcloud.size();
}

struct FitLineResidual {
    template<typename T>
    bool operator() (const T* first_point,
                     const T* second_point,
                     T* residuals) const {
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      // Cast everything over to generic
      const Vector2T first_pointT(first_point[0], first_point[1]);
      const Vector2T second_pointT(second_point[0], second_point[1]);
      const Vector2T pointT = point.cast<T>();
      //Vector2T center = center_of_mass.cast<T>();
      Vector2T line_seg_start = line_seg.start_point.cast<T>();
      Vector2T line_seg_end = line_seg.end_point.cast<T>();
      // Find the centroid (missing denominator which is used later).
      T r = T((center_of_mass - line_seg.start_point).norm() +
            (center_of_mass - line_seg.end_point).norm());
      Vector2T diff_vec = second_pointT - first_pointT;
      T t = (pointT - first_pointT).dot(diff_vec) /
            (diff_vec.dot(diff_vec));
      T dist;
      if (t < T(0)) {
        dist = (pointT - line_seg_start).dot(pointT - line_seg_start);
      } else if (t > T(1)) {
        dist = (pointT - line_seg_end).dot(pointT - line_seg_end);
      } else {
        //dist = (pointT - first_pointT + t * (second_pointT - first_pointT)).norm();
        Vector2T proj = line_seg_start + t * (diff_vec);
        dist = (pointT - proj).dot(pointT - proj);
      }
      residuals[0] = (r / T(point_num)) + dist;
      return true;
    }

    FitLineResidual(const LineSegment& line_seg,
                    const Vector2f point,
                    const Vector2f center_of_mass,
                    const size_t point_num) :
                    line_seg(line_seg),
                    point(point),
                    center_of_mass(center_of_mass),
                    point_num(point_num) {}

    static AutoDiffCostFunction<FitLineResidual, 1, 2, 2>*
    create(const LineSegment& line_seg,
           const Vector2f point,
           const Vector2f center_of_mass,
           const size_t point_num) {
      FitLineResidual *line_res = new FitLineResidual(line_seg, point, center_of_mass, point_num);
      return new AutoDiffCostFunction<FitLineResidual, 1, 2, 2>(line_res);
    }

    const LineSegment line_seg;
    const Vector2f point;
    const Vector2f center_of_mass;
    const size_t point_num;
};

LineSegment FitLine(LineSegment line, const vector<Vector2f> pointcloud) {
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  ceres::Problem problem;
  double first_point[] = {line.start_point.x(), line.start_point.y()};
  double second_point[] = {line.end_point.x(), line.end_point.y()};
  Vector2f center_of_mass = GetCenterOfMass(pointcloud);
  size_t point_num = pointcloud.size();
  for (const Vector2f& point : pointcloud) {
    problem.AddResidualBlock(FitLineResidual::create(line,
                                                     point,
                                                     center_of_mass,
                                                     point_num),
                             NULL,
                             first_point,
                             second_point);
  }
  ceres::Solve(options, &problem, &summary);
  Vector2f new_start_point(first_point[0], first_point[1]);
  Vector2f new_end_point(second_point[0], second_point[1]);
  return LineSegment(new_start_point, new_end_point);
}

vector<Vector2f> GetPointsFromInliers(vector<pair<int, Vector2f>> inliers) {
  vector<Vector2f> points;
  for (const pair<int, Vector2f> p : inliers) {
    points.push_back(p.second);
  }
  return points;
}

vector<LineSegment> ExtractLines(const vector <Vector2f>& pointcloud) {
  if (pointcloud.size() <= 1) {
    return vector<LineSegment>();
  }
  vector<Vector2f> remaining_points = pointcloud;
  vector<LineSegment> lines;
  while (remaining_points.size() > 1) {
    std::cout << "Remaining Points " << remaining_points.size() << "/" << pointcloud.size() << std::endl;
    // Restrict the RANSAC implementation to using a small subset of the points.
    // This will speed it up.
    vector<Vector2f> neighborhood = GetNeighborhood(remaining_points);
    if (neighborhood.size() <= 0) {
      break;
    }
    LineSegment line = RANSACLineSegment(neighborhood);
    LineSegment new_line = FitLine(line, neighborhood);
    vector<pair<int, Vector2f>> inliers;
    do {
      inliers = GetInliers(new_line, remaining_points);
      line = new_line;
      new_line = FitLine(line, GetPointsFromInliers(inliers));
      // Test if we get more inliers from increasing neighborhood
      LineSegment test_line = FitLine(new_line, GetNeighborhoodAroundLine(new_line, remaining_points));
      if (GetInliers(test_line, remaining_points).size() >= GetInliers(new_line, remaining_points).size()) {
        new_line = test_line;
      }
    } while ((new_line.start_point - line.start_point).norm() +
             (new_line.end_point - line.end_point).norm() > CONVERGANCE_THRESHOLD);
    lines.push_back(new_line);
    // We have to remove the points that were assigned to this line.
    // Sort the inliers by their index so we don't get weird index problems.
    inliers = GetInliers(new_line, remaining_points);
    sort(inliers.begin(), inliers.end(), [](pair<int, Vector2f> p1,
                                            pair<int, Vector2f> p2) {
      return p1.first < p2.first;
    });
    if (inliers.size() < 2) {
      continue;
    }
    for (int64_t i = inliers.size() - 1; i >= 0; i--) {
      remaining_points.erase(remaining_points.begin() + inliers[i].first);
    }
  }
  std::cout << "Pointcloud size: " << pointcloud.size() << std::endl;
  std::cout << "Lines size: " << lines.size() << std::endl;
  return lines;
}
//
//Eigen::Matrix2f EstimateCovarianceFromPoints(const vector<Vector2f>& points) {
//  Eigen::Matrix2f scatter_matrix = Eigen::Matrix2f::Zero();
//  // Find means for covariance calculation
//  double x_mean = 0.0;
//  double y_mean = 0.0;
//  for (const Vector2f& p : points) {
//    x_mean += p.x();
//    y_mean += p.y();
//  }
//  x_mean /= points.size();
//  y_mean /= points.size();
//  // Find Covariances for each entry in the matrix.
//  for (const Vector2f& p : points) {
//    scatter_matrix(0, 0) += (p.x() - x_mean) * (p.x() - x_mean);
//    scatter_matrix(0, 1) += (p.x() - x_mean) * (p.y() - y_mean);
//    scatter_matrix(1, 0) += (p.y() - y_mean) * (p.x() - x_mean);
//    scatter_matrix(1, 1) += (p.y() - y_mean) * (p.y() - y_mean);
//  }
//  Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
//  covariance(0, 0) = scatter_matrix(0, 0) / (points.size() - 1);
//  covariance(0, 1) = scatter_matrix(0, 1) / (points.size() - 1);
//  covariance(1, 0) = scatter_matrix(1, 0) / (points.size() - 1);
//  covariance(1, 1) = scatter_matrix(1, 1) / (points.size() - 1);
//  return covariance;
//}

/*
 * Returns point sampled from a gaussian around this source_point,
 * characterized by the covariance.
 */
//Vector2f Sample(const Vector2f source_point, Eigen::Matrix2f covariance) {
//  std::default_random_engine gen;
//  std::normal_distribution x_dist(source_point.x(), covariance(0, 0));
//  std::normal_distribution y_dist(source_point.y(), covariance(1, 1));
//  double rand_x = x_dist(gen);
//  double rand_y = y_dist(gen);
//  return Vector2f(rand_x, rand_y);
//}
//
//Eigen::Matrix2f GetSensorCovariance(const Vector2f& point, const SensorCovParams& cov_params) {
//  // Assumes that pointclouds are centered at (0,0).
//  double range = point.norm();
//  // Assumes that pointclouds are oriented towards (1, 0) always.
//  Vector2f viewpoint(1, 0);
//  double angle = acos((point / range).dot(viewpoint));
//  Eigen::Matrix2f sensor_covariance;
//  sensor_covariance << 2 * (sin(angle) * sin(angle)), -sin(2 * angle), -sin(2 * angle), 2 * (cos(angle) * cos(angle));
//  double scaling_factor_1 = (range * range * cov_params.std_dev_angle * cov_params.std_dev_angle) / 2;
//  sensor_covariance = scaling_factor_1 * sensor_covariance;
//  Eigen::Matrix2f mat_2;
//  mat_2 << cos(angle) * cos(angle), sin(2 * angle), sin(2 * angle), 2 * sin(angle) * sin(angle);
//  double scaling_factor_2 = cov_params.std_dev_range / 2;
//  mat_2 = scaling_factor_2 * mat_2;
//  return sensor_covariance + mat_2;
//}
//
//vector<LineCovariances> GetLineEndpointCovariances(const vector<LineSegment> lines,
//                                                   const vector<Vector2f>& points,
//                                                   const SensorCovParams& sensor_cov_params) {
//  vector<LineCovariances> line_endpoint_covariances;
//  for (LineSegment line : lines) {
//    vector<Vector2f> sample_start_points;
//    vector<Vector2f> sample_end_points;
//    vector<Vector2f> inliers = GetPointsFromInliers(GetInliers(line, points));
//    for (uint64_t iter = 0; iter < UNCERTAINTY_SAMPLE_NUM; iter++) {
//      vector<Vector2f> sampled_points;
//      for (const Vector2f& point : inliers) {
//        const Eigen::Matrix2f sensor_cov = GetSensorCovariance(point, sensor_cov_params);
//        const Vector2f sampled_point = Sample(point, sample_point, sensor_cov);
//        sampled_points.push_back(sampled_point)
//      }
//      LineSegment sampled_line = FitLine(RANSACLineSegment(sampled_points), sampled_points);
//      sample_start_points.push_back(sampled_line.start_point);
//      sample_start_points.push_back(sampled_line.end_point);
//    }
//    Eigen::Matrix3f start_point_cov = EstimateCovarianceFromPoints(sample_start_points);
//    Eigen::Matrix3f end_point_cov = EstimateCovarianceFromPoints(sample_end_points);
//    line_endpoint_covariances.emplace_back(start_point_cov, end_point_cov);
//  }
//}


}