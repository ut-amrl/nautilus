//
// Created by jack on 9/13/20.
//

#ifndef FEATURE_IDENT_FEATURE_EXTRACTER_H
#define FEATURE_IDENT_FEATURE_EXTRACTER_H

#include <Eigen/Dense>

namespace nautilus::input_processing {
// Implements the feature extraction used in the LOAM system paper:
// https://ri.cmu.edu/pub_files/2014/7/Ji_LidarMapping_RSS2014_v8.pdf
// Does not use the sub region splitting at the current time.

class FeatureExtractor {
 public:
  FeatureExtractor(const std::vector<Eigen::Vector2f>& points, double threshold,
                   double distance_threshold, int neighbor_num,
                   int max_edge_num, int max_planar_num, int min_neighbor_num);
  std::vector<Eigen::Vector2f> GetPlanarPoints();
  std::vector<Eigen::Vector2f> GetEdgePoints();

 private:
  std::vector<float> GetSmoothnessScores();

  double threshold_ = 0.008;
  double distance_threshold_;
  double max_neighbor_distance_ = 0.8;
  int neighbors_per_side_ = 10;
  size_t max_edge_number_ = 2;
  size_t max_planar_number_ = 4;
  int min_neighbor_num_ = 3;

  std::vector<std::pair<double, Eigen::Vector2f>> smoothness_points_;
  std::vector<float> unordered_scores_;
  std::vector<int> planar_points_;
  std::vector<int> edge_points_;
};
}  // namespace nautilus::input_processing

#endif  // FEATURE_IDENT_FEATURE_EXTRACTER_H
