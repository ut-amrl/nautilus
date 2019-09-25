//
// Created by jack on 9/15/19.
//

#ifndef ICP_ICP_H
#define ICP_ICP_H

using Eigen::Vector2d;


std::vector<Vector2d> PerformIcp(std::vector<Vector2d> source_points,
                                 PointCloud2 &source_points_m,
                                 std::vector<Vector2d> target_points,
                                 PointCloud2 &target_points_m,
                                 bool debug);

#endif //ICP_ICP_H
