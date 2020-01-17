#include <cstdlib>
#include <time.h>
#include <iostream>
#include <vector>
#include <unordered_map>

#include "Eigen/Dense"

#include "./normal_computation.h"
#include "./kdtree.h"

/* Normal computation is based on this paper
 * http://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf
 * Fast and Robust Normal Estimation for Point Cloudswith Sharp Features
 * By Boulch et al.
 */

using std::vector;
using NormalComputation::CircularHoughAccumulator;

#define NEIGHBORHOOD_SIZE 0.15
#define NEIGHBORHOOD_STEP_SIZE 0.1
#define MEAN_DISTANCE 0.1
#define LIMIT_CORRECT_PROBABILITY 0.95
#define BIN_NUMBER 32

inline Vector2f GetNormalFromAngle(double angle) {
  Eigen::Matrix2f rot = Eigen::Rotation2Df(angle).toRotationMatrix();
  return rot * Vector2f(1, 0);
}

inline size_t SampleLimit(size_t bin_number) {
  return (1 / (2.0 * MEAN_DISTANCE * MEAN_DISTANCE));
}

inline double BinMean(size_t bin_num, 
                      const CircularHoughAccumulator& accum) {
  return accum.Votes(bin_num) / accum.accumulator.size();
}

inline bool MeansDontIntersect(CircularHoughAccumulator& accum) {
  double mean_difference =
    BinMean(accum.GetMostVotedBinIndex(), accum) - 
    BinMean(accum.GetSecondMostVotedBinIndex(), accum);
  // Assuming confidence level of 95%
  double lower_bound = 2.0 * sqrt(1.0 / accum.accumulator.size());
  return mean_difference >= lower_bound;
}

vector<Vector2f>
NormalComputation::GetNormals(const vector<Vector2f>& points) {
  // For each point we have to randomly sample points within its neighborhood.
  // Then when we either reach the upper limit of samples, or pass the 
  // threshold of confidence and stop.
  // Pick the most selected bin and set the normal of this point to the average
  // angle of that bin.
  // Compute the line using that angle and a point at (1,0).
  srand(time(NULL));
  KDTree<float, 2>* tree =
    new KDTree<float, 2>(KDTree<float, 2>::EigenToKDNoNormals(points));
  vector<Vector2f> normals;
  for (const Vector2f& point : points) {
    CircularHoughAccumulator accum(BIN_NUMBER);
    size_t number_of_samples = 0;
    double neighborhood_size = NEIGHBORHOOD_SIZE;
    vector<KDNodeValue<float, 2>> neighbors;
    while (neighbors.size() <= 1) {
      tree->FindNeighborPoints(point, neighborhood_size, &neighbors); 
      neighborhood_size += NEIGHBORHOOD_STEP_SIZE;
    }
    std::unordered_map<size_t, bool> chosen_samples;
    // Check that the sample limit is less than total number of choices.
    size_t limit = std::min(neighbors.size() * (neighbors.size() - 1), SampleLimit(BIN_NUMBER));
    std::cout << "Neighborhood size: " << neighbors.size() << std::endl;
    std::cout << "Limit: " << limit << std::endl;
    std::cout << "Sample Limit: " << SampleLimit(BIN_NUMBER) << std::endl;
    while (number_of_samples < limit) { 
      // Get a random pair of points using a costless combinatorial ordering.
      size_t first_index;
      size_t second_index;
      do {
        first_index = rand() % neighbors.size();
        second_index = rand() % neighbors.size();
      } while (first_index == second_index || chosen_samples[neighbors.size() * first_index + second_index]);
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
    normals.push_back(
      GetNormalFromAngle(
        accum.AverageBinAngle(
          accum.GetMostVotedBinIndex())));
  }
  return normals;
}
