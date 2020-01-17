#ifndef SRC_NORMAL_COMPUTATION_H
#define SRC_NORMAL_COMPUTATION_H

#include <vector>

#include "Eigen/Dense"

#include "./math_util.h"


using std::vector;
using Eigen::Vector2f;
using math_util::angle_mod;

namespace NormalComputation {
  struct CircularHoughAccumulator {
    vector<vector<double>> accumulator;
    const double angle_step;
    size_t most_voted_bin;
    size_t second_most_voted_bin;
  
    // The bin number is the number of bins around the equator.
    CircularHoughAccumulator(size_t bin_number) : 
      accumulator(bin_number, vector<double>()), 
      angle_step(M_2PI / bin_number),
      most_voted_bin(0),
      second_most_voted_bin(0) {}
  
    inline size_t Votes(size_t bin_number) const {
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
  
    size_t GetMostVotedBinIndex() const {
      return most_voted_bin;
    }
  
    size_t GetSecondMostVotedBinIndex() const {
      return second_most_voted_bin;
    }
  
    double AverageBinAngle(size_t bin_number) const {
      double avg = 0.0;
      for (const double angle : accumulator[bin_number]) {
        avg += angle;
      }
      return avg / Votes(bin_number);
    }
  };

  // Returns a list of normals for the corresponding list of points. 
  // Normals are unit sized, but because we use meters as our unit, 
  // they are 1 meter long.
  vector<Vector2f> GetNormals(const vector<Vector2f>& points);
}

#endif
