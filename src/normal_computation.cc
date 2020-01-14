#include <cstdlib>
#include <time.h>
#include <vector>

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

#define NEIGHBORHOOD_SIZE 0.25
#define NEIGHBORHOOD_STEP_SIZE 0.1
#define MEAN_DISTANCE 0.1
#define LIMIT_CORRECT_PROBABILITY 0.95
#define BIN_NUMBER 32

inline Vector2f GetNormalFromAngle(double angle) {
  Eigen::Matrix2f rot = Eigen::Rotation2Df(angle).getRotationMatrix();
  return rot * Vector2f(1, 0);
}

inline size_t SampleLimit(size_t bin_number) {
  return (1 / (2.0 * MEAN_DISTANCE * MEAN_DISTANCE))
}

inline double BinMean(size_t bin_num, 
                      const CircularHoughAccumulator& accum) {
  return accum.votes(bin_num) / accum.accumulator.size();
}

inline bool MeansDontIntersect(CircularHoughAccumulator& accum) {
  double mean_difference =
    BinMean(accum.GetMostVotedBinIndex(), accum) - 
    BinMean(accum.GetSecondMostVotedBinIndex(), acumm);
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
  const KDTree<float, 2>* tree =
    new KDTree<float, 2>(KDTree<2, float>::EigenToKD(points));
  vector<Vector2f> normals;
  for (const Vector2f& point : points) {
    CircularHoughAccumulator accum(BIN_NUMBER);
    size_t number_of_samples = 0;
    double neighborhood_size = NEIGHBORHOOD_SIZE;
    vector<KDNodeValue<float, 2> neighbors;
    while (neighbors.size() <= 1) {
      FindNeighborPoints(point, neighborhood_size, &neighbors); 
      neighborhood_size += NEIGHBORHOOD_STEP_SIZE;
    }
    while (number_of_samples < SampleLimit(BIN_NUMBER)) { 
      // Get a random pair of points using a costless combinatorial ordering. 
      // TODO: Costless Combinatorial Ordering
      Vector2f point_1 = neighbors[rand() % neighbors.size()];
      Vector2f point_2;
      do {
        point_2 = neighbors[rand() % neighbors.size()];
      } while (point_2.x() == point_1.x() && point_2.y() == point_1.y());
      // Now come up with their normal.
      Eigen::Hyperplane<float, 2> surface_line =
        Eigen::Hyperplane<float, 2>::Through(point_1, point_2);
      Vector2f normal = surface_line.normal();
      double angle_with_x_axis = acos(normal.dot(Vector2f(1, 0)));
      accum.AddVote(angle_with_x_axis);
      if (MeansDontIntersect(accum)) {
        break; 
      }
    }
    return GetNormalFromAngle(AverageBinAngle(GetMostVotedBinIndex()));
  }
}
