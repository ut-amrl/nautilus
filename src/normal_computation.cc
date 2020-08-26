#include "./normal_computation.h"

#include <time.h>

#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "./kdtree.h"
#include "./math_util.h"
#include "Eigen/Dense"

/* Normal computation is based on this paper
 * http://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf
 * Fast and Robust Normal Estimation for Point Clouds with Sharp Features
 * By Boulch et al.
 */

namespace nautilus::NormalComputation {

using Eigen::Rotation2Df;
using Eigen::Vector2f;
using std::vector;

CONFIG_DOUBLE(neighborhood_size, "nc_neighborhood_size");
CONFIG_DOUBLE(neighborhood_step_size, "nc_neighborhood_step_size");
CONFIG_DOUBLE(mean_distance, "nc_mean_distance");
CONFIG_INT(bin_number, "nc_bin_number");

// TODO: Possibly adaptively set the k neighborhood size.
// TODO: Orientation of normals.

inline Vector2f GetNormalFromAngle(double angle) {
  Eigen::Matrix2f rot = Eigen::Rotation2Df(angle).toRotationMatrix();
  return rot * Vector2f(1, 0);
}

inline size_t SampleLimit(double mean_distance) {
  return (1 / (2.0 * mean_distance * mean_distance));
}

inline double BinMean(size_t bin_num, const CircularHoughAccumulator &accum) {
  return accum.Votes(bin_num) / accum.accumulator.size();
}

inline bool MeansDontIntersect(CircularHoughAccumulator &accum) {
  double mean_difference = BinMean(accum.GetMostVotedBinIndex(), accum) -
                           BinMean(accum.GetSecondMostVotedBinIndex(), accum);
  // Assuming confidence level of 95%
  double lower_bound = 2.0 * sqrt(1.0 / accum.accumulator.size());
  return mean_difference >= lower_bound;
}

vector<double> LargestClusterWithinThreshold(const vector<double> normal_angles,
                                             double threshold) {
  // Find the largest cluster, or the normal with the most similar normals.
  vector<double> largest_cluster;
  for (size_t i = 0; i < normal_angles.size(); i++) {
    vector<double> temp_cluster;
    for (size_t j = 0; j < normal_angles.size(); j++) {
      if (fabs(normal_angles[i] - normal_angles[j]) <= threshold) {
        temp_cluster.push_back(normal_angles[j]);
      }
    }
    if (temp_cluster.size() > largest_cluster.size()) {
      largest_cluster = temp_cluster;
    }
  }
  return largest_cluster;
}

vector<Vector2f> GetNormals(const vector<Vector2f> &points) {
  // For each point we have to randomly sample points within its neighborhood.
  // Then when we either reach the upper limit of samples, or pass the
  // threshold of confidence and stop.
  // Pick the most selected bin and set the normal of this point to the average
  // angle of that bin.
  // Compute the line using that angle and a point at (1,0).
  srand(time(NULL));
  KDTree<float, 2> *tree =
      new KDTree<float, 2>(KDTree<float, 2>::EigenToKDNoNormals(points));
  vector<Vector2f> normals;
  for (const Vector2f& point : points) {
    CircularHoughAccumulator accum(NormalComputationConfig::CONFIG_bin_number);
    size_t number_of_samples = 0;
    double neighborhood_size = NormalComputationConfig::CONFIG_neighborhood_size;
    vector<KDNodeValue<float, 2>> neighbors;
    while (neighbors.size() <= 1) {
      tree->FindNeighborPoints(point, neighborhood_size, &neighbors);
      neighborhood_size += NormalComputationConfig::CONFIG_neighborhood_step_size;
    }
    std::unordered_map<size_t, bool> chosen_samples;
    // Check that the sample limit is less than total number of choices.
    size_t limit = std::min(neighbors.size() * (neighbors.size() - 1),
                            SampleLimit(NormalComputationConfig::CONFIG_mean_distance));
    while (number_of_samples < limit) {
      // Get a random pair of points using a costless combinatorial ordering.
      size_t first_index;
      size_t second_index;
      do {
        first_index = rand() % neighbors.size();
        second_index = rand() % neighbors.size();
      } while (first_index == second_index ||
               chosen_samples[neighbors.size() * first_index + second_index]);
      chosen_samples[neighbors.size() * first_index + second_index] = true;
      Vector2f point_1 = neighbors[first_index].point;
      Vector2f point_2 = neighbors[second_index].point;
      // Now come up with their normal.
      Eigen::Hyperplane<float, 2> surface_line =
          Eigen::Hyperplane<float, 2>::Through(point_1, point_2);
      Vector2f normal = surface_line.normal();
      double angle_with_x_axis = acos(normal.dot(Vector2f(1, 0)));
      accum.AddVote(angle_with_x_axis);
      if (MeansDontIntersect(accum)) {
        break;
      }
      number_of_samples++;
    }
    normals.push_back(GetNormalFromAngle(
        accum.AverageBinAngle(accum.GetMostVotedBinIndex())));
  }
  return normals;
}

}  // namespace nautilus
