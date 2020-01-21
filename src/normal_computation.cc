#include <cstdlib>
#include <time.h>
#include <iostream>
#include <vector>
#include <unordered_map>

#include "Eigen/Dense"

#include "./normal_computation.h"
#include "./kdtree.h"
#include "./math_util.h"

/* Normal computation is based on this paper
 * http://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf
 * Fast and Robust Normal Estimation for Point Cloudswith Sharp Features
 * By Boulch et al.
 */

using std::vector;
using NormalComputation::CircularHoughAccumulator;
using Eigen::Rotation2Df;

#define NEIGHBORHOOD_SIZE 0.15
#define NEIGHBORHOOD_STEP_SIZE 0.1
#define MEAN_DISTANCE 0.1
#define LIMIT_CORRECT_PROBABILITY 0.95
#define BIN_NUMBER 32
#define CLUSTERING_THRESHOLD M_PI / 26.0

// TODO: We could adaptively select the k neighborhood size. But, too difficult to do by the meeting.

#define K_NEIGHBORHOOD 10


//struct RiemannEdge {
//  size_t node_i;
//  size_t node_j;
//  double edge_cost;
//
//  RiemannEdge(size_t node_i,
//              size_t node_j,
//              Vector2f node_i_normal,
//              Vector2f node_j_normal) 
//              : node_i(node_i), 
//                node_j(node_j) {
//    edge_cost = 1 - fabs(node_i_normal.dot(node_j_normal));
//  }
//}
//
//vector<Vector2f> GetKNeighborhood(const Vector2f point, const KDTree* kd_tree) {
//  double threshold = 0.05;
//  vector<KDNodeValue<float, 2>> neighbors;
//  do {
//    neighbors.clear();
//    kd_tree->FindNeighborPoints(point, threshold, &neighbors);
//    // Sort by distance to neighbors.
//    sort(neighbors.begin(),
//         neighbors.end(),
//         [&point](KDNodeValue<float, 2> p1,
//                  KDNodeValue<float, 2> p2) { 
//         return (point - p1).norm() < (point - p2).norm()});
//    threshold += 0.1;
//  } while (neighbors.size() < K_NEIGHBORHOOD);
//  vector<Vector2f> k_neighborhood;
//  for (const KDNodeValue& neighbor : neighbors) {
//    k_neighborhood.push_back(neighbor);
//  }
//  return k_neighborhood;
//}
//
//
//struct RiemannGraph {
//  vector<RiemannEdge> edges;
//
//  // TODO : We need to construct the graph
//  // TODO: Then get the MST
//  // TODO: Then finally loop over the MST and invert the normals as we go if needed.
//
//  RiemannGraph(vector<Vector2f> pointcloud,
//               vector<Vector2f> normals) {
//    CHECK_GE(pointcloud.size(), K_NEIGHBORHOOD);
//    vector<RiemannEdge> possible_edges;
//    KDTree<float, 2> pointcloud_tree =
//      new KDTree<float, 2>(KDTree<float, 2>::EigenToKDNoNormals(pointcloud));
//    for (const Vector2f& point : pointcloud) {
//      vector<Vector2f> neighborhood = 
//        GetKNeighborhood(point, pointcloud_tree);
//      
//    }
//  }
//}

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

Vector2f MostLikelyNormal(const vector<double> normals) {
  double angle_accum = 0.0;
  vector<double> largest_cluster =
    LargestClusterWithinThreshold(normals, CLUSTERING_THRESHOLD);
  for(const double angle : largest_cluster) {
    angle_accum += angle;
  }
  return GetNormalFromAngle(angle_accum / largest_cluster.size());
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
        neighbors[second_index].point;
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
    normals.push_back(GetNormalFromAngle(accum.AverageBinAngle(accum.GetMostVotedBinIndex())));
  }
  return normals;
}
