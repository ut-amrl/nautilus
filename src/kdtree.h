
//========================================================================
//  This software is free: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License Version 3,
//  as published by the Free Software Foundation.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  Version 3 in the file COPYING that came with this distribution.
//  If not, see <http://www.gnu.org/licenses/>.
//========================================================================
// Copyright 2012 joydeepb@ri.cmu.edu
// Robotics Institute, Carnegie Mellon University
//
// A general K-Dimension Tree (KDTree) implementation.

#ifndef SRC_KDTREE_H_
#define SRC_KDTREE_H_

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <ctime>

#include "config_reader/config_reader.h"
#include "math_util.h"

namespace NormalComputation {
    template <typename T, unsigned int K>
    inline std::vector<Eigen::Matrix<T, K, 1>> GetNormals(const std::vector<Eigen::Matrix<T, K, 1>>& points);
}

template <typename T, unsigned int K>
struct KDNodeValue {
  Eigen::Matrix<T, K, 1> point;
  Eigen::Matrix<T, K, 1> normal;
  int index;
  KDNodeValue() : index(0) {}
  KDNodeValue(const Eigen::Matrix<T, K, 1>& _point,
              const Eigen::Matrix<T, K, 1>& _normal, int _index)
      : point(_point), normal(_normal), index(_index) {}
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T, unsigned int K>
class KDTree {
 public:
  // Default constructor: Creates an empty KDTree with uninitialized root node.
  KDTree()
      : splitting_dimension_(0),
        left_tree_(nullptr),
        right_tree_(nullptr),
        parent_tree_(nullptr) {}

  // Disallow the copy constructor.
  KDTree(const KDTree<T, K>& values) = delete;

  ~KDTree() {
    delete left_tree_;
    delete right_tree_;
  }

  // Disallow the assignment operator.
  KDTree<T, K>& operator=(const KDTree<T, K>& other) = delete;

  // Construct the KDTree using the @values provided.
  explicit KDTree(const std::vector<KDNodeValue<T, K>>& values)
          : splitting_dimension_(0),
            left_tree_(nullptr),
            right_tree_(nullptr),
            parent_tree_(nullptr) {
    BuildKDTree(values);
  }

  // Construct the KDTree using the @points provided, except eigen based values.
  explicit KDTree(const std::vector<Eigen::Matrix<T, K, 1>>& points, bool compute_normals=true)
          : splitting_dimension_(0),
            left_tree_(nullptr),
            right_tree_(nullptr),
            parent_tree_(nullptr) {
    if (compute_normals) {
      EigenToKD(points);
    } else {
      EigenToKDNoNormals(points);
    }
  }

  // Rebuild the KDTree using the @values provided.
  KDTree* BuildKDTree(std::vector<KDNodeValue<T, K>> values) {
    if (values.size() == 0) {
      // Return an empty tree;
      left_tree_ = nullptr;
      right_tree_ = nullptr;
      return this;
    }

    // Determine splitting plane.
    splitting_dimension_ = GetSplittingPlane(values);
    VectorComparator comparator(splitting_dimension_);

    // Sort the points in order of their distance from the splitting plane.
    std::sort(values.begin(), values.end(), comparator);

    // Take the median point and make that the root.
    unsigned int ind = values.size() / 2;
    // Make sure this is the first of this value in this dimension, as we want the right tree to be
    // greater than or equal.
    double epsilon = 0.0001;
    while (ind > 0 &&
          (values[ind].point(splitting_dimension_) - values[ind - 1].point(splitting_dimension_)) < epsilon) {
      ind--;
    }

    value_ = std::shared_ptr<KDNodeValue<T, K>>(new KDNodeValue<T, K>(values[ind].point, values[ind].normal, values[ind].index));
//    CHECK(sizeof(*value_.get()) % 16 == 0);

    // Insert the KD tree of the left hand part of the list to the left node.
    left_tree_ = NULL;
    if (ind > 0) {
      std::vector<KDNodeValue<T, K>> points_left(values.begin(), values.begin() + ind);
      left_tree_ = new KDTree<T,K>(points_left);
      left_tree_->parent_tree_ = this;
    }

    // Insert the KD tree of the right hand part of the list to the right node.
    right_tree_ = NULL;
    if (ind < values.size() - 1) {
      std::vector<KDNodeValue<T, K>> points_right(values.begin() + ind + 1,
                                             values.end());
      right_tree_ = new KDTree<T, K>(points_right);
      right_tree_->parent_tree_ = this;
    }
    return this;
  }

  void EigenToKD(const std::vector<Eigen::Matrix<T, K, 1>>& values) {
    std::vector<KDNodeValue<T, K>> kd_values;
    CHECK_GE(values.size(), 0);
    auto normals = NormalComputation::GetNormals<T, K>(values);
    for (size_t node_index = 0; node_index < values.size(); node_index++) {
      kd_values.emplace_back(values[node_index], normals[node_index],
                             node_index);
    }
    BuildKDTree(kd_values);
  }

  void EigenToKDNoNormals(const std::vector<Eigen::Matrix<T, K, 1>>& values) {
    std::vector<KDNodeValue<T, K>> point_nodes;
    CHECK_GE(values.size(), 0);
    for (size_t node_index = 0; node_index < values.size(); node_index++) {
      Eigen::Matrix<T, K, 1> zero_normal = Eigen::Matrix<T, K, 1>::Zero();
      point_nodes.emplace_back(values[node_index], zero_normal, node_index);
    }
    BuildKDTree(point_nodes);
  }

  // Finds the nearest point and normal in the KDTree to the provided &point.
  // Returns the distance to the nearest neighbor if one is found within the
  // specified threshold. Distance from the nearest neighbor along the
  // associated normal is used as the distance metric. This is useful for
  // (for example) point to plane ICP.
  T FindNearestPointNormal(const Eigen::Matrix<T, K, 1>& point,
                                         const T& threshold,
                                         std::shared_ptr<KDNodeValue<T, K>> neighbor_node) const {
    if (value_ == nullptr) {
      return threshold;
    }
    T current_best_dist = FLT_MAX;
    if ((value_->point - point).squaredNorm() < threshold * threshold) {
      neighbor_node = value_;
      current_best_dist = fabs(value_->normal.dot(point - value_->point));
      if (current_best_dist < FLT_MIN) {
        return T(0.0);
      }
    }

    // The signed distance of the point from the splitting plane.
    const T point_distance(point(splitting_dimension_) -
                           value_->point(splitting_dimension_));

    // Follow the query point down the tree and return the best neighbor down
    // that branch.
    KDTree* other_tree = nullptr;
    if (point_distance < 0.0 && left_tree_ != nullptr) {
      std::shared_ptr<KDNodeValue<T, K>> left_tree_neighbor_node;
      T left_tree_dist = left_tree_->FindNearestPointNormal(
              point, threshold, left_tree_neighbor_node);
      if (left_tree_dist < current_best_dist) {
        current_best_dist = left_tree_dist;
        neighbor_node = left_tree_neighbor_node;
      }
      other_tree = right_tree_;
    }
    if (point_distance >= 0.0 && right_tree_ != nullptr) {
      std::shared_ptr<KDNodeValue<T, K>> right_tree_neighbor_node;
      T right_tree_dist = right_tree_->FindNearestPointNormal(
              point, threshold, right_tree_neighbor_node);
      if (right_tree_dist < current_best_dist) {
        current_best_dist = right_tree_dist;
        neighbor_node = right_tree_neighbor_node;
      }
      other_tree = left_tree_;
    }

    // Check if the point is closer to the splitting plane than the current
    // best neighbor. If so, the other side needs to be checked as well.
    if (other_tree != nullptr && point_distance != 0.0 &&
        fabs(point_distance) < min(current_best_dist, threshold)) {
      std::shared_ptr<KDNodeValue<T, K>> other_tree_neighbor_node;
      T other_tree_dist = other_tree->FindNearestPointNormal(
              point, threshold, other_tree_neighbor_node);
      if (other_tree_dist < current_best_dist) {
        current_best_dist = other_tree_dist;
        neighbor_node = other_tree_neighbor_node;
      }
    }

    return current_best_dist;
  }

  // Finds the set of points in the KDTree closer than the distance &threshold
  // to the provided &point. Euclidean L2 norm is used as the distance metric
  // for nearest neighbor search.
  void FindNeighborPoints(
          const Eigen::Matrix<T, K, 1>& point, const T& threshold,
          std::vector<std::shared_ptr<KDNodeValue<T, K>>>* neighbor_points) {
    if (value_ == nullptr) {
      return;
    }
    T current_dist = (value_->point - point).norm();
    if (current_dist < threshold) neighbor_points->push_back(value_);

    // The signed distance of the point from the splitting plane.
    const T point_distance(point(splitting_dimension_) -
                           value_->point(splitting_dimension_));

    if (point_distance < threshold && left_tree_ != NULL) {
      left_tree_->FindNeighborPoints(point, threshold, neighbor_points);
    }

    if (point_distance > -threshold && right_tree_ != NULL) {
      right_tree_->FindNeighborPoints(point, threshold, neighbor_points);
    }
  }

  // Finds the nearest point in the KDTree to the provided &point. Returns
  // the distance to the nearest neighbor found if one is found within the
  // specified threshold. Euclidean L2 norm is used as the distance metric for
  // nearest neighbor search. This is useful for (for example) point to
  // point ICP.
  T FindNearestPoint(const Eigen::Matrix<T, K, 1>& point,
                                   const T& threshold,
                                   std::shared_ptr<KDNodeValue<T, K>>& neighbor_node) const {
    if (value_ == nullptr) {
      return threshold;
    }
    T current_best_dist = (value_->point - point).norm();
    neighbor_node = value_;
    if (current_best_dist < FLT_MIN) {
      return T(0.0);
    }

    // The signed distance of the point from the splitting plane.
    const T point_distance(point(splitting_dimension_) -
                           value_->point(splitting_dimension_));

    // Follow the query point down the tree and return the best neighbor down
    // that branch.
    KDTree* other_tree = nullptr;
    if (point_distance < 0.0 && left_tree_ != nullptr) {
      std::shared_ptr<KDNodeValue<T, K>> left_tree_neighbor_node;
      const T left_tree_dist = left_tree_->FindNearestPoint(
              point, min(current_best_dist, threshold), left_tree_neighbor_node);
      if (left_tree_dist < current_best_dist) {
        current_best_dist = left_tree_dist;
        neighbor_node = left_tree_neighbor_node;
      }
      other_tree = right_tree_;
    }
    if (point_distance >= 0.0 && right_tree_ != nullptr) {
      std::shared_ptr<KDNodeValue<T, K>> right_tree_neighbor_node;
      const T right_tree_dist = right_tree_->FindNearestPoint(
              point, min(current_best_dist, threshold), right_tree_neighbor_node);
      if (right_tree_dist < current_best_dist) {
        current_best_dist = right_tree_dist;
        neighbor_node = right_tree_neighbor_node;
      }
      other_tree = left_tree_;
    }

    // Check if the point is closer to the splitting plane than the current
    // best neighbor. If so, the other side needs to be checked as well.
    if (other_tree != nullptr && point_distance != 0.0 &&
        fabs(point_distance) < min(current_best_dist, threshold)) {
      std::shared_ptr<KDNodeValue<T, K>> other_tree_neighbor_node;
      const T other_tree_dist = other_tree->FindNearestPoint(
              point, min(current_best_dist, threshold), other_tree_neighbor_node);
      if (other_tree_dist < current_best_dist) {
        current_best_dist = other_tree_dist;
        neighbor_node = other_tree_neighbor_node;
      }
    }

    return current_best_dist;
  }

  // Removes the first point that is within the threshold from the specified point.
  // Returns the new tree with that point removed. Does not mean that is reconstructing the whole tree.
  // The threshold should be small so that not more than one point falls within the sphere around the point with radius
  // threshold, otherwise you may have the wrong point removed.
  void RemoveNearestValue(const Eigen::Matrix<T, K, 1>& point, const T& threshold) {
    if (value_ == nullptr) {
      return;
    }
    // This is the node we are looking for.
    // TODO: What if threshold is really large.
    if ((point - value_->point).norm() < threshold) {
      if (left_tree_ == nullptr && right_tree_ == nullptr) {
        // Delete from parent node.
        if (parent_tree_ != nullptr) {
          if (parent_tree_->left_tree_ != nullptr && parent_tree_->left_tree_->value_->index == value_->index) {
            parent_tree_->left_tree_ = nullptr;
            delete this; // No longer accessible, this is ok.
          } else if (parent_tree_->right_tree_ != nullptr &&
                     parent_tree_->right_tree_->value_->index == value_->index) {
            parent_tree_->right_tree_ = nullptr;
            delete this; // No longer accessible, this is ok.
          }
        } else {
          // This is a leaf node, free it and set it to nullptr.
          value_ = nullptr;
        }
        return;
      }
      // If there only is a missing right tree, then we need to find the largest in the left tree.
      if (right_tree_ == nullptr) {
        // If we are missing the right tree then find the largest point in this splitting dimension in the
        // left tree.
        std::shared_ptr<KDNodeValue<T, K>> found_node = left_tree_->GetMaximumValue(splitting_dimension_);
        value_ = found_node;
        // Delete this other node.
        left_tree_->RemoveNearestValue(found_node->point, threshold);
      } else {
        // If we are missing the left tree or have both then find the minimum point in the right sub-tree.
        std::shared_ptr<KDNodeValue<T, K>> found_node = right_tree_->GetMinimumValue(splitting_dimension_);
        value_ = found_node;
        // Delete that found_point elsewhere in the tree.
        right_tree_->RemoveNearestValue(found_node->point, threshold);
      }
      return;
    }
    // Distance along the splitting plane. If negative go left, otherwise go right.
    T splitting_plane_distance = point(splitting_dimension_) - value_->point(splitting_dimension_);
    if (splitting_plane_distance < 0 && left_tree_ != nullptr) {
      // Check the left sub-tree and set the result to be the tree with the point removed.
      left_tree_->RemoveNearestValue(point, threshold);
    } else if (splitting_plane_distance >= 0 && right_tree_ != nullptr) {
      // Check the right sub-tree and set the result to be the tree with the point removed.
      right_tree_->RemoveNearestValue(point, threshold);
    }
  }

  std::shared_ptr<KDNodeValue<T, K>> GetMinimumValue(int splitting_dimension) {
    // Check wasn't called on an invalid tree.
    if (value_ == nullptr) {
      return nullptr;
    }
    // Three cases.
    // 1. Leaf node, return itself.
    // 3. One Child:
    //      a. If same splitting dim.
    //          i. Only go down it if its the left, if doesn't exist return the current value.
    //      b. Different splitting dim.
    //          i. Go down it and check if it's value is lower than the current value at this node.
    // 2. Two children:
    //      a. If same splitting dim then we go left.
    //      b. Otherwise, both directions. Return the smallest.

    // Case 1, Leaf Node.
    if (left_tree_ == nullptr && right_tree_ == nullptr) {
      return value_;
    }
    // Case 2, One Child.
    if (left_tree_ == nullptr || right_tree_ == nullptr) {
      // a. Same splitting dim.
      if (splitting_dimension == splitting_dimension_) {
        // If there is a left, go down. Otherwise return the current.
        if (left_tree_ != nullptr) {
          return left_tree_->GetMinimumValue(splitting_dimension);
        } else {
          return value_;
        }
      }
      // b. Different splitting dim.
      KDTree* non_null_sub_tree = (left_tree_ == nullptr)? right_tree_ : left_tree_;
      std::shared_ptr<KDNodeValue<T, K>> sub_tree_minimum = non_null_sub_tree->GetMinimumValue(splitting_dimension);
      if (sub_tree_minimum->point(splitting_dimension) < value_->point(splitting_dimension)) {
        return sub_tree_minimum;
      }
      return value_;
    }
    // Case 3, Two Children.
    CHECK(left_tree_ != nullptr && right_tree_ != nullptr);
    // If the same splitting dim.
    if (splitting_dimension == splitting_dimension_) {
      return left_tree_->GetMinimumValue(splitting_dimension);
    }
    // Otherwise, both directions and return the smallest of this node, and the sub-tree mins.
    std::shared_ptr<KDNodeValue<T, K>> left_min = left_tree_->GetMinimumValue(splitting_dimension);
    std::shared_ptr<KDNodeValue<T, K>> right_min = right_tree_->GetMinimumValue(splitting_dimension);
    // If the left min is the smallest in this splitting dimension.
    if (left_min->point(splitting_dimension) < value_->point(splitting_dimension) &&
        left_min->point(splitting_dimension) < right_min->point(splitting_dimension)) {
      return left_min;
    }
    // If the right min is the smallest in this splitting dimension.
    if (right_min->point(splitting_dimension) < value_->point(splitting_dimension) &&
        right_min->point(splitting_dimension) < left_min->point(splitting_dimension)) {
      return right_min;
    }
    // Lastly then, value must be the smallest.
    return value_;
  }

  std::shared_ptr<KDNodeValue<T, K>> GetMaximumValue(const int splitting_dimension) {
    // Check wasn't called on an invalid tree.
    if (value_ == nullptr) {
      return nullptr;
    }
    // Three cases.
    // 1. Leaf node, return itself.
    // 3. One Child:
    //      a. If same splitting dim.
    //          i. Only go down it if its the right, if doesn't exist return the current value.
    //      b. Different splitting dim.
    //          i. Go down it and check if it's value is higher than the current value at this node.
    // 2. Two children:
    //      a. If same splitting dim then we go right.
    //      b. Otherwise, both directions. Return the largest.

    // Case 1, Leaf Node.
    if (left_tree_ == nullptr && right_tree_ == nullptr) {
      return value_;
    }
    // Case 2, One Child.
    if (left_tree_ == nullptr || right_tree_ == nullptr) {
      // a. Same splitting dim.
      if (splitting_dimension == splitting_dimension_) {
        // If there is a left, go down. Otherwise return the current.
        if (right_tree_ != nullptr) {
          return right_tree_->GetMaximumValue(splitting_dimension);
        } else {
          return value_;
        }
      }
      // b. Different splitting dim.
      KDTree* non_null_sub_tree = (left_tree_ == nullptr)? right_tree_ : left_tree_;
      std::shared_ptr<KDNodeValue<T, K>> sub_tree_minimum = non_null_sub_tree->GetMaximumValue(splitting_dimension);
      if (sub_tree_minimum->point(splitting_dimension) > value_->point(splitting_dimension)) {
        return sub_tree_minimum;
      }
      return value_;
    }
    // Case 3, Two Children.
    CHECK(left_tree_ != nullptr && right_tree_ != nullptr);
    // If the same splitting dim.
    if (splitting_dimension == splitting_dimension_) {
      return right_tree_->GetMaximumValue(splitting_dimension);
    }
    // Otherwise, both directions and return the largest of this node, and the sub-tree maxs.
    std::shared_ptr<KDNodeValue<T, K>> left_min = left_tree_->GetMaximumValue(splitting_dimension);
    std::shared_ptr<KDNodeValue<T, K>> right_min = right_tree_->GetMaximumValue(splitting_dimension);
    // If the left max is the largest in this splitting dimension.
    if (left_min->point(splitting_dimension) > value_->point(splitting_dimension) &&
        left_min->point(splitting_dimension) > right_min->point(splitting_dimension)) {
      return left_min;
    }
    // If the right max is the largest in this splitting dimension.
    if (right_min->point(splitting_dimension) > value_->point(splitting_dimension) &&
        right_min->point(splitting_dimension) > left_min->point(splitting_dimension)) {
      return right_min;
    }
    // Lastly then, value must be the largest.
    return value_;
  }

private:

  inline T min(const T& a, const T& b) const {
    return (a < b) ? a : b;
  }

  int GetSplittingPlane(const std::vector<KDNodeValue<T, K>>& values) {
    Eigen::Matrix<T, K, 1> mean;
    Eigen::Matrix<T, K, 1> std_dev;
    mean.setZero();
    std_dev.setZero();

    // Compute mean along all dimensions.
    for (unsigned int i = 0; i < values.size(); ++i) {
      mean = mean + values[i].point;
    }
    mean = mean / static_cast<T>(values.size());

    // Compute standard deviation along all dimensions.
    for (unsigned int i = 0; i < values.size(); ++i) {
      for (unsigned int j = 0; j < K; ++j) {
        std_dev(j) = std_dev(j) + (values[i].point(j) - mean(j)) *
                                  (values[i].point(j) - mean(j));
      }
    }

    // Chose the splitting plane along the dimension that has the greatest spread,
    // as indicated by the standard deviation along that dimension.
    int splitting_plane = 0;
    T max_std_dev(0.0);
    for (unsigned int j = 0; j < K; ++j) {
      if (std_dev(j) > max_std_dev) {
        splitting_plane = j;
        max_std_dev = std_dev(j);
      }
    }
    return splitting_plane;
  }

  // Comparator functor used for sorting points based on their values along a
  // particular dimension.
  struct VectorComparator {
      const unsigned int comparator_dimension;
      explicit VectorComparator(int dimension) : comparator_dimension(dimension) {}
      bool operator()(const KDNodeValue<T, K>& v1, const KDNodeValue<T, K>& v2) {
        return v1.point(comparator_dimension) < v2.point(comparator_dimension);
      }
  };

  // The dimension along which the split is, as this node.
  int splitting_dimension_;

  std::shared_ptr<KDNodeValue<T, K>> value_;

  KDTree* left_tree_;
  KDTree* right_tree_;
  KDTree* parent_tree_;
};

/* Normal computation is based on this paper
 * http://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf
 * Fast and Robust Normal Estimation for Point Clouds with Sharp Features
 * By Boulch et al.
 */

namespace NormalComputation {

    using Eigen::Vector2f;
    using math_util::angle_mod;
    using std::vector;
    using Eigen::Rotation2Df;

    CONFIG_DOUBLE(neighborhood_size, "nc_neighborhood_size");
    CONFIG_DOUBLE(neighborhood_step_size, "nc_neighborhood_step_size");
    CONFIG_DOUBLE(mean_distance, "nc_mean_distance");
    CONFIG_INT(bin_number, "nc_bin_number");

    struct CircularHoughAccumulator {
        vector<vector<double>> accumulator;
        const double angle_step;
        size_t most_voted_bin;
        size_t second_most_voted_bin;

        // The bin number is the number of bins around the equator.
        CircularHoughAccumulator(size_t bin_number)
                : accumulator(bin_number, vector<double>()),
                  angle_step(M_2PI / bin_number),
                  most_voted_bin(0),
                  second_most_voted_bin(0) {}

        size_t Votes(size_t bin_number) const {
          return accumulator[bin_number].size();
        }

        void AddVote(double angle_in_radians) {
          angle_in_radians = angle_mod(angle_in_radians);
          size_t bin_number = round(angle_in_radians / angle_step);
          if (bin_number > accumulator.size()) {
            return;
          }
          accumulator[bin_number].push_back(angle_in_radians);
          if (Votes(most_voted_bin) < Votes(bin_number)) {
            second_most_voted_bin = most_voted_bin;
            most_voted_bin = bin_number;
          } else if (Votes(second_most_voted_bin) < Votes(bin_number)) {
            second_most_voted_bin = bin_number;
          }
        }

        size_t GetMostVotedBinIndex() const { return most_voted_bin; }

        size_t GetSecondMostVotedBinIndex() const { return second_most_voted_bin; }

        double AverageBinAngle(size_t bin_number) const {
          double avg = 0.0;
          for (const double angle : accumulator[bin_number]) {
            avg += angle;
          }
          return avg / Votes(bin_number);
        }

        double BinMean(size_t bin_num) {
          return Votes(bin_num) / accumulator.size();
        }

        bool MeansDontIntersect() {
          double mean_difference = BinMean(GetMostVotedBinIndex()) -
                                   BinMean(GetSecondMostVotedBinIndex());
          // Assuming confidence level of 95%
          double lower_bound = 2.0 * sqrt(1.0 / accumulator.size());
          return mean_difference >= lower_bound;
        }
    };

    inline size_t SampleLimit(double mean_distance) {
      return (1 / (2.0 * mean_distance * mean_distance));
    }

    inline Vector2f GetNormalFromAngle(double angle) {
      Eigen::Matrix2f rot = Eigen::Rotation2Df(angle).toRotationMatrix();
      return rot * Vector2f(1, 0);
    }

  template <typename T, unsigned int K>
  inline std::vector<Eigen::Matrix<T, K, 1>> GetNormals(const std::vector<Eigen::Matrix<T, K, 1>>& points) {
      typedef Eigen::Matrix<T, K, 1> VectorKT;
      if (CONFIG_neighborhood_size <= 0) {
        throw std::runtime_error("Invalid neighborhood_size for normal computation");
      }
      // For each point we have to randomly sample points within its neighborhood.
      // Then when we either reach the upper limit of samples, or pass the
      // threshold of confidence and stop.
      // Pick the most selected bin and set the normal of this point to the average
      // angle of that bin.
      // Compute the line using that angle and a point at (1,0).
      srand(time(NULL));
      auto* tree = new KDTree<T, K>(points, false);
      vector<VectorKT> normals;
      for (const VectorKT& point : points) {
        CircularHoughAccumulator accum(CONFIG_bin_number);
        size_t number_of_samples = 0;
        double neighborhood_size = CONFIG_neighborhood_size;
        vector<std::shared_ptr<KDNodeValue<T, K>>> neighbors;
        while (neighbors.size() <= 1) {
          tree->FindNeighborPoints(point, neighborhood_size, &neighbors);
          neighborhood_size += CONFIG_neighborhood_step_size;
        }
        std::unordered_map<size_t, bool> chosen_samples;
        // Check that the sample limit is less than total number of choices.
        size_t limit = std::min(neighbors.size() * (neighbors.size() - 1),
                                SampleLimit(CONFIG_mean_distance));
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
          VectorKT point_1 = neighbors[first_index]->point;
          VectorKT point_2 = neighbors[second_index]->point;
          // Now come up with their normal.
          Eigen::Hyperplane<T, K> surface_line =
                  Eigen::Hyperplane<T, K>::Through(point_1, point_2);
          VectorKT normal = surface_line.normal();
          VectorKT x_axis = VectorKT::Zero();
          x_axis(0) = 1;
          double angle_with_x_axis = acos(normal.dot(x_axis));
          accum.AddVote(angle_with_x_axis);
          if (accum.MeansDontIntersect()) {
            break;
          }
          number_of_samples++;
        }
        normals.push_back(GetNormalFromAngle(
                accum.AverageBinAngle(accum.GetMostVotedBinIndex())));
      }
      delete tree;
      return normals;
    }
}

#endif  // SRC_KDTREE_H_
