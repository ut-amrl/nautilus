//
// Created by jack on 9/25/19.
//

#include "./solver.h"

#include <omp.h>
#include <utility>
#include <queue>
#include <algorithm>
#include <queue>
#include <thread>
#include <vector>

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"

#include "./kdtree.h"
#include "./slam_types.h"
#include "./math_util.h"
#include "./pointcloud_helpers.h"
#include "./gui_helpers.h"
#include "timer.h"
#include "lidar_slam/HitlSlamInputMsg.h"

#define LIDAR_CONSTRAINT_AMOUNT 5 //TODO: Change this back to 10 after testing.
#define OUTLIER_THRESHOLD 0.25
#define HITL_LINE_WIDTH 0.05
#define HITL_POSE_POINT_THRESHOLD 5

using std::vector;
using slam_types::OdometryFactor2D;
using slam_types::LidarFactor;
using ceres::AutoDiffCostFunction;
using Eigen::Matrix2f;
using Eigen::Vector2f;
using Eigen::Rotation2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using Eigen::Affine2f;
using lidar_slam::HitlSlamInputMsgConstPtr;
using lidar_slam::HitlSlamInputMsg;
using slam_types::SLAMNode2D;

visualization_msgs::Marker match_line_list;
visualization_msgs::Marker normals_marker;

template<typename T> Eigen::Transform<T, 2, Eigen::Affine>
PoseArrayToAffine(const T* rotation, const T* translation) {
  typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
  typedef Eigen::Rotation2D<T> Rotation2DT;
  typedef Eigen::Translation<T, 2> Translation2T;
  Affine2T affine = Translation2T(translation[0], translation[1]) *
    Rotation2DT(rotation[0]).toRotationMatrix();
  return affine;
}

struct OdometryResidual {
    template <typename T>
    bool operator() (const T* pose_i,
                     const T* pose_j,
                     T* residual) const {
      // Predicted pose_j = pose_i * odometry.
      // Hence, error = pose_j.inverse() * pose_i * odometry;
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      typedef Eigen::Matrix<T, 2, 2> Matrix2T;
      // Extract the rotation matrices.
      const Matrix2T Ri = Rotation2D<T>(pose_i[2])
        .toRotationMatrix();
      const Matrix2T Rj = Rotation2D<T>(pose_j[2])
        .toRotationMatrix();
      // Extract the translation.
      const Vector2T Ti(pose_i[0], pose_i[1]);
      const Vector2T Tj(pose_j[0], pose_j[1]);
      // The Error in the translation is the difference with the odometry
      // in the direction of the previous pose, then getting rid of the new 
      // rotation (transpose = inverse for rotation matrices).
      const Vector2T error_translation =
        Rj.transpose() * (Ri * T_odom.cast<T>() - (Tj - Ti));
      // Rotation error is very similar to the translation error, except
      // we don't care about the difference in the position.
      const Matrix2T error_rotation_mat =
        Rj.transpose() * Ri * R_odom.cast<T>();
      // The residuals are weighted according to the parameters set
      // by the user.
      residual[0] = T(translation_weight) * error_translation.x();
      residual[1] = T(translation_weight) * error_translation.y();
      residual[2] = T(rotation_weight) *
        Rotation2D<T>().fromRotationMatrix(error_rotation_mat).angle();
      return true;
    }

    OdometryResidual(const OdometryFactor2D& factor,
                     double translation_weight,
                     double rotation_weight) :
            translation_weight(translation_weight),
            rotation_weight(rotation_weight),
            R_odom(Rotation2D<float>(factor.rotation)
                    .toRotationMatrix()),
            T_odom(factor.translation) {}

    static AutoDiffCostFunction<OdometryResidual, 3, 3, 3>*
    create(const OdometryFactor2D& factor,
           double translation_weight,
           double rotation_weight) {
      OdometryResidual* residual = new OdometryResidual(factor,
                                                        translation_weight,
                                                        rotation_weight);
      return new AutoDiffCostFunction<OdometryResidual, 3, 3, 3>(residual);
    }

    double translation_weight;
    double rotation_weight;
    const Matrix2f R_odom;
    const Vector2f T_odom;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

struct LIDARPointBlobResidual {

    // TODO: Add the source normals penalization as well.
    // Would cause there to be two normals.
    template <typename T>
    bool operator() (const T* source_pose,
                     const T* target_pose,
                     T* residuals) const {
      typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      const Affine2T source_to_world =
        PoseArrayToAffine(&source_pose[2], &source_pose[0]);
      const Affine2T world_to_target =
        PoseArrayToAffine(&target_pose[2], &target_pose[0]).inverse();
      const Affine2T source_to_target = world_to_target * source_to_world;
      #pragma omp parallel for default(none) shared(residuals)
      for (size_t index = 0; index < source_points.size(); index++) {
        Vector2T source_pointT = source_points[index].cast<T>();
        Vector2T target_pointT = target_points[index].cast<T>();
        // Transform source_point into the frame of target_point
        source_pointT = source_to_target * source_pointT;
        T target_normal_result =
          target_normals[index].cast<T>().dot(source_pointT - target_pointT);
        T source_normal_result =
          source_normals[index].cast<T>().dot(target_pointT - source_pointT);
        residuals[index * 2] = target_normal_result;
        residuals[index * 2 + 1] = source_normal_result;
      }
      return true;
    }

    LIDARPointBlobResidual(vector<Vector2f>& source_points,
                           vector<Vector2f>& target_points,
                           vector<Vector2f>& source_normals,
                           vector<Vector2f>& target_normals) :
                           source_points(source_points),
                           target_points(target_points),
                           source_normals(source_normals),
                           target_normals(target_normals) {
      CHECK_EQ(source_points.size(), target_points.size());
      CHECK_EQ(target_points.size(), target_normals.size());
      CHECK_EQ(source_normals.size(), target_normals.size());
    }

    static AutoDiffCostFunction<LIDARPointBlobResidual, ceres::DYNAMIC, 3, 3>*
    create(vector<Vector2f>& source_points,
           vector<Vector2f>& target_points,
           vector<Vector2f>& source_normals,
           vector<Vector2f>& target_normals) {
      LIDARPointBlobResidual *residual =
        new LIDARPointBlobResidual(source_points,
                                   target_points,
                                   source_normals,
                                   target_normals);
      return new AutoDiffCostFunction<LIDARPointBlobResidual,
                                      ceres::DYNAMIC,
                                      3, 3>
                                      (residual, source_points.size() * 2);
    }

    const vector<Vector2f> source_points;
    const vector<Vector2f> target_points;
    const vector<Vector2f> source_normals;
    const vector<Vector2f> target_normals;
};

class VisualizationCallback : public ceres::IterationCallback {
 public:
  VisualizationCallback(const SLAMProblem2D& problem,
                        const vector<SLAMNodeSolution2D>* solution,
                        ros::NodeHandle& n) :
                        problem(problem),
                        solution(solution) {
    pointcloud_helpers::InitPointcloud(&all_points_marker);
    pointcloud_helpers::InitPointcloud(&new_points_marker);
    all_points.clear();
    point_pub = n.advertise<sensor_msgs::PointCloud2>("/all_points", 10);
    new_point_pub = n.advertise<sensor_msgs::PointCloud2>("/new_points", 10);
    pose_pub = n.advertise<visualization_msgs::Marker>("/poses", 10);
    match_pub = n.advertise<visualization_msgs::Marker>("/matches", 10);
    normals_pub = n.advertise<visualization_msgs::Marker>("/normals", 10);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_STRIP,
                                  gui_helpers::Color4f::kGreen,
                                  0.002,
                                  0.0,
                                  0.0,
                                  &pose_array);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kBlue,
                                  0.003,
                                  0.0,
                                  0.0,
                                  &match_line_list);
    gui_helpers::InitializeMarker(visualization_msgs::Marker::LINE_LIST,
                                  gui_helpers::Color4f::kYellow,
                                  0.003,
                                  0.0,
                                  0.0,
                                  &normals_marker);
    }

  void PubVisualization() {
    const vector<SLAMNodeSolution2D>& solution_c = *solution;
    vector<Vector2f> new_points;
    for (size_t i = 0; i < solution_c.size(); i++) {
      if (new_points.size() > 0) {
        all_points.insert(all_points.end(),
                          new_points.begin(),
                          new_points.end());
        new_points.clear();
      }
      auto pointcloud = problem.nodes[i].lidar_factor.pointcloud;
      Affine2f robot_to_world =
        PoseArrayToAffine(&(solution_c[i].pose[2]),
                          &(solution_c[i].pose[0])).cast<float>();
      Eigen::Vector3f pose(solution_c[i].pose[0], solution_c[i].pose[1], 0.0);
      gui_helpers::AddPoint(pose, gui_helpers::Color4f::kGreen, &pose_array);
      for (const Vector2f& point : pointcloud) {
        new_points.push_back(robot_to_world * point);
        // Visualize normal
        KDNodeValue<float, 2> source_point_in_tree;
        float dist =
          problem.nodes[i].lidar_factor
            .pointcloud_tree->FindNearestPoint(point,
                                               0.01,
                                               &source_point_in_tree);
        if (dist != 0.01) {
          Eigen::Vector3f normal(source_point_in_tree.normal.x(),
                                 source_point_in_tree.normal.y(),
                                 0.0);
          normal = robot_to_world * normal;
          Vector2f source_point = robot_to_world * point;
          Eigen::Vector3f source_3f(source_point.x(), source_point.y(), 0.0);
          Eigen::Vector3f result = source_3f + (normal * 0.01);
          gui_helpers::AddLine(source_3f,
                               result,
                               gui_helpers::Color4f::kGreen,
                               &normals_marker);
        }
      }
    }
    if (solution_c.size() >= 2) {
      pointcloud_helpers::PublishPointcloud(all_points,
                                            all_points_marker,
                                            point_pub);
      pointcloud_helpers::PublishPointcloud(new_points,
                                            new_points_marker,
                                            new_point_pub);
      pose_pub.publish(pose_array);
      match_pub.publish(match_line_list);
      normals_pub.publish(normals_marker);
    }
    ros::spinOnce();
    all_points.clear();
    gui_helpers::ClearMarker(&pose_array);
  }

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
  override {
    PubVisualization();
    return ceres::SOLVER_CONTINUE;
  }

 private:
  sensor_msgs::PointCloud2 all_points_marker;
  sensor_msgs::PointCloud2 new_points_marker;
  std::vector<Vector2f> all_points;
  const SLAMProblem2D& problem;
  const vector<SLAMNodeSolution2D>* solution;
  ros::Publisher point_pub;
  ros::Publisher pose_pub;
  ros::Publisher match_pub;
  ros::Publisher new_point_pub;
  ros::Publisher normals_pub;
  visualization_msgs::Marker pose_array;
};

void Solver::AddOdomFactors(ceres::Problem* ceres_problem,
                            double trans_weight,
                            double rot_weight) {
  for (const OdometryFactor2D& odom_factor : problem_.odometry_factors) {
    CHECK_LT(odom_factor.pose_i, odom_factor.pose_j);
    CHECK_GT(solution_.size(), odom_factor.pose_i);
    CHECK_GT(solution_.size(), odom_factor.pose_j);
    ceres_problem->AddResidualBlock(
      OdometryResidual::create(odom_factor, trans_weight, rot_weight),
      NULL,
      solution_[odom_factor.pose_i].pose,
      solution_[odom_factor.pose_j].pose);
  }
}

inline bool NormalsSimilar(const Vector2f& n1,
                           const Vector2f& n2,
                           float max_cosine_value) {
  return (fabs(n1.dot(n2)) > max_cosine_value);
}

// Source moves to target.
double
Solver::GetPointCorrespondences(const SLAMProblem2D& problem,
                                vector<SLAMNodeSolution2D>* solution_ptr,
                                PointCorrespondences* point_correspondences,
                                size_t source_node_index,
                                size_t target_node_index) {
  // Summed differences between point correspondences.
  double difference = 0.0;
  vector<SLAMNodeSolution2D> &solution = *solution_ptr;
  SLAMNodeSolution2D &source_solution = solution[source_node_index];
  SLAMNodeSolution2D &target_solution = solution[target_node_index];
  LidarFactor source_lidar =
          problem.nodes[source_node_index].lidar_factor;
  LidarFactor target_lidar =
          problem.nodes[target_node_index].lidar_factor;
  // Affine transformations from the two pose's reference frames.
  Affine2f source_to_world =
          PoseArrayToAffine(&source_solution.pose[2],
                            &source_solution.pose[0]).cast<float>();
  Affine2f target_to_world =
          PoseArrayToAffine(&target_solution.pose[2],
                            &target_solution.pose[0]).cast<float>();
  // Loop over all the points in the source pointcloud,
  // match each point to the closest point in the target pointcloud
  // who's normal is within a certain threshold.
  for (const Vector2f &source_point : source_lidar.pointcloud) {
    // Transform the source point to the target frame.
    Vector2f source_point_transformed =
            target_to_world.inverse() * source_to_world * source_point;
    // Get the closest points within the threshold.
    // For now we assume that a match is within 1/6 of the threshold.
    KDNodeValue<float, 2> closest_target;
    vector<KDNodeValue<float, 2>> neighbors;
    target_lidar.pointcloud_tree->FindNeighborPoints(source_point_transformed,
                                                     OUTLIER_THRESHOLD / 6.0,
                                                     &neighbors);
    // Get the current source point's normal.
    KDNodeValue<float, 2> source_point_with_normal;
    float found_dist =
            source_lidar.pointcloud_tree->FindNearestPoint(source_point,
                                                           0.1,
                                                           &source_point_with_normal);
    CHECK_EQ(found_dist, 0.0) << "Source point is not in KD Tree!\n";
    float dist = OUTLIER_THRESHOLD;
    // Sort the target points by distance from the source point in the
    // target frame.
    std::sort(neighbors.begin(),
              neighbors.end(),
              [&source_point_transformed](KDNodeValue<float, 2> point_1,
                                          KDNodeValue<float, 2> point_2) {
                  return (source_point_transformed - point_1.point).norm() <
                         (source_point_transformed - point_2.point).norm();
              });
    // For all target points, starting with the closest
    // see if any of them have a close enough normal to be 
    // considered a match.
    for (KDNodeValue<float, 2> current_target : neighbors) {
      if (NormalsSimilar(current_target.normal,
                         source_point_with_normal.normal,
                         cos(math_util::DegToRad(20.0)))) {
        closest_target = current_target;
        dist = (source_point_transformed - current_target.point).norm();
        break;
      }
    }
    // If we didn't find any matches in the first 1/6 of the threshold,
    // try all target points within the full threshold.
    if (dist >= OUTLIER_THRESHOLD) {
      // Re-find all the closest targets.
      neighbors.clear();
      target_lidar.pointcloud_tree->FindNeighborPoints(source_point_transformed,
                                                       OUTLIER_THRESHOLD,
                                                       &neighbors);
      // Sort them again, based on distance from the source point in the
      // target frame.
      std::sort(neighbors.begin(),
                neighbors.end(),
                [&source_point_transformed](KDNodeValue<float, 2> point_1,
                                            KDNodeValue<float, 2> point_2) {
                    return (source_point_transformed - point_1.point).norm() <
                           (source_point_transformed - point_2.point).norm();
                });
      // Cut out the first 1/6 threshold that we already checked.
      vector<KDNodeValue<float, 2>>
              unchecked_neighbors(neighbors.begin() + (OUTLIER_THRESHOLD / 6),
                                  neighbors.end());
      // See if any of these points have a normal within our threshold.
      for (KDNodeValue<float, 2> current_target : unchecked_neighbors) {
        if (NormalsSimilar(current_target.normal,
                           source_point_with_normal.normal,
                           cos(math_util::DegToRad(20.0)))) {
          closest_target = current_target;
          dist = (source_point_transformed - current_target.point).norm();
          break;
        }
      }
      // If no target point was found to correspond to our source point then
      // don't match this source point to anything.
      if (dist >= OUTLIER_THRESHOLD) {
        continue;
      }
    }
    // Add the distance between the source point and it's matching target
    // point.
    difference += dist;
    // Add to the correspondence for returning.
    Vector2f source_point_modifiable = source_point;
    point_correspondences->source_points.push_back(source_point_modifiable);
    point_correspondences->target_points.push_back(closest_target.point);
    point_correspondences->
            source_normals.push_back(source_point_with_normal.normal);
    point_correspondences->target_normals.push_back(closest_target.normal);
    // Add a line from the matches that we are using.
    // Transform everything to the world frame.
    // This is for the visualization.
    source_point_transformed = target_to_world * source_point_transformed;
    Vector2f closest_point_in_target = target_to_world * closest_target.point;
    Eigen::Vector3f source_3d(source_point_transformed.x(),
                              source_point_transformed.y(), 0.0f);
    Eigen::Vector3f target_3d(closest_point_in_target.x(),
                              closest_point_in_target.y(), 0.0f);
    gui_helpers::AddLine(source_3d,
                         target_3d,
                         gui_helpers::Color4f::kBlue,
                         &match_line_list);
  }
  return difference;
}

double Solver::AddLidarResidualsForLC(ceres::Problem& problem) {
  double difference = 0.0;
  for (const LCConstraint constraint : loop_closure_constraints_) {
    for (const LCPose& a_pose : constraint.line_a_poses) {
      for (const LCPose& b_pose : constraint.line_b_poses) {
        CHECK_LT(a_pose.node_idx, solution_.size());
        CHECK_LT(b_pose.node_idx, solution_.size());
        CHECK_LT(a_pose.node_idx, problem_.nodes.size());
        CHECK_LT(b_pose.node_idx, problem_.nodes.size());
        PointCorrespondences correspondence(solution_[a_pose.node_idx].pose,
                                            solution_[b_pose.node_idx].pose);
        difference += GetPointCorrespondences(problem_,
                                              &solution_,
                                              &correspondence,
                                              a_pose.node_idx,
                                              b_pose.node_idx);
        difference /=
                problem_.nodes[a_pose.node_idx].lidar_factor.pointcloud.size();
        // Add the correspondences as constraints in the optimization problem.
        problem.AddResidualBlock(
                LIDARPointBlobResidual::create(correspondence.source_points,
                                               correspondence.target_points,
                                               correspondence.source_normals,
                                               correspondence.target_normals),
                NULL,
                correspondence.source_pose,
                correspondence.target_pose);
      }
    }
  }
  return difference;
}

vector<SLAMNodeSolution2D>
Solver::SolveSLAM() {
  // Setup ceres for evaluation of the problem.
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = std::thread::hardware_concurrency();
  VisualizationCallback vis_callback(problem_, &solution_, n_);
  options.callbacks.push_back(&vis_callback);
  double difference = 0;
  double last_difference = 0;
  // While our solution moves more than the stopping_accuracy,
  // continue to optimize.
  for (int64_t window_size = 1;
       window_size <= LIDAR_CONSTRAINT_AMOUNT;
       window_size++) {
    LOG(INFO) << "Using window size: " << window_size << std::endl;
    do {
      gui_helpers::ClearMarker(&match_line_list);
      gui_helpers::ClearMarker(&normals_marker);
      last_difference = difference;
      difference = 0;
      ceres::Problem ceres_problem;
      // Add all the odometry constraints between our poses.
      AddOdomFactors(&ceres_problem,
                     translation_weight_,
                     rotation_weight_);
      // For every SLAM node we want to optimize it against the past
      // LIDAR_CONSTRAINT_AMOUNT nodes.
      for (size_t node_i_index = 0;
           node_i_index < problem_.nodes.size();
           node_i_index++) {
        // Set the first pose to be constant.
        if (node_i_index == 0) {
          ceres_problem.SetParameterBlockConstant(solution_[0].pose);
        }
        for (size_t node_j_index =
                std::max((int64_t) (node_i_index) - window_size,
                         0l);
             node_j_index < node_i_index;
             node_j_index++) {
          PointCorrespondences correspondence(solution_[node_j_index].pose,
                                              solution_[node_i_index].pose);
          // Get the correspondences between these two poses.
          difference += GetPointCorrespondences(problem_,
                                                &solution_,
                                                &correspondence,
                                                node_j_index,
                                                node_i_index);
          difference /=
            problem_.nodes[node_j_index].lidar_factor.pointcloud.size();
          // Add the correspondences as constraints in the optimization problem.
          ceres_problem.AddResidualBlock(
                  LIDARPointBlobResidual::create(correspondence.source_points,
                                                 correspondence.target_points,
                                                 correspondence.source_normals,
                                                 correspondence.target_normals),
                  NULL,
                  correspondence.source_pose,
                  correspondence.target_pose);
        }
      }
      difference += AddLidarResidualsForLC(ceres_problem);
      AddColinearResiduals(&ceres_problem);
      ceres::Solve(options, &ceres_problem, &summary);
    } while (abs(difference - last_difference) > stopping_accuracy_);
  }
  // Call the visualization once more to see the finished optimization.
  for (int i = 0; i < 5; i++) {
    vis_callback.PubVisualization();
    sleep(1);
  }
  return solution_;
}

Solver::Solver(double translation_weight,
               double rotation_weight,
               double lc_translation_weight,
               double lc_rotation_weight,
               double stopping_accuracy,
               SLAMProblem2D& problem,
               ros::NodeHandle& n) :
               translation_weight_(translation_weight),
               rotation_weight_(rotation_weight),
               lc_translation_weight_(lc_translation_weight),
               lc_rotation_weight_(lc_rotation_weight),
               stopping_accuracy_(stopping_accuracy),
               problem_(problem),
               n_(n) {
  // Copy all the data to a list that we are going to modify as we optimize.
  for (size_t i = 0; i < problem.nodes.size(); i++) {
    // Make sure that we marked all the data correctly earlier.
    CHECK_EQ(i, problem.nodes[i].node_idx);
    SLAMNodeSolution2D sol_node(problem.nodes[i]);
    solution_.push_back(sol_node);
  }
  CHECK_EQ(solution_.size(), problem.nodes.size());
}

/*
 * Methods relevant to HITL loop closure.
 */

template <typename T>
T DistanceToLineSegment(const Eigen::Matrix<T, 2, 1>& point,
                             const LineSegment<T>& line_seg) {
  typedef Eigen::Matrix<T, 2, 1> Vector2T;
  // Line segment is parametric, with a start point and endpoint.
  // Parameterized by t between 0 and 1.
  // We can get the point on the line by projecting the start -> point onto
  // this line.
  Eigen::Hyperplane<T, 2> line =
    Eigen::Hyperplane<T, 2>::Through(line_seg.start, line_seg.endpoint);
  Eigen::Hyperplane<T, 2> start_to_point =
    Eigen::Hyperplane<T, 2>::Through(line_seg.start, point);
  Vector2T point_on_line = line.projection(point);
  T line_length = (line_seg.endpoint - line_seg.start).norm();
  T t = (point_on_line - line_seg.start).norm() / line_length;
  if (t >= T(0.0) && t <= T(1.0)) {
    // Point is between start and end, should return perpendicular dist.
    return line.absDistance(point);
  }
  // Point is closer to an endpoint.
  return std::min<T>((line_seg.start - point).norm(),
                  (line_seg.endpoint - point).norm());
}

Vector2f GetCenterOfMass(vector<Vector2f> pointcloud) {
  Vector2f total;
  for (Vector2f point : pointcloud) {
    total += point;
  }
  return total / pointcloud.size();
}

template <typename T>
T ObservationSumCost(const LCPose& pose,
                     LineSegment<T> line_seg,
                     Eigen::Transform<T, 2, 2, Eigen::Affine>& pose_to_world) {
  typedef Eigen::Matrix<T, 2, 1> Vector2T;
  T sum = T(0.0);
  for (const Vector2f point : pose.points_on_feature) {
    Vector2T pointT = point.cast<T>();
    pointT = pose_to_world * pointT;
    sum += DistanceToLineSegment<T>(pointT, line_seg);
  }
  sum /= T(pose.points_on_feature.size());
  sum = pow(sum, 0.5);
  return sum;
}

struct ColinearResidual {
    template <typename T>
    bool operator() (const T* pose_a,
                     const T* pose_b,
                     T* residuals) const {
      typedef Eigen::Transform<T, 2, 2, Eigen::Affine> Affine2T;
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      // From Human-In-The-Loop SLAM (Nashed et al, 2017)
      // --- R_a & R_b Residual ---
      // Sum of observations distance to the feature (the line) divided by the
      // size of the observations, raised to the one half.
      Affine2T a_to_world = PoseArrayToAffine(&pose_a[0], &pose_a[2]);
      Affine2T b_to_world = PoseArrayToAffine(&pose_b[0], &pose_b[2]);
      residuals[0] = ObservationSumCost<T>(pose_a_constraint,
                                           line_a.cast<T>(),
                                           a_to_world);
      residuals[1] = ObservationSumCost<T>(pose_b_constraint,
                                           line_b.cast<T>(),
                                           b_to_world);
      // --- R_p Residual ---
      // K_1 * norm((cm_b - cm_a) * n_a) + K_2 * (1 - (n_a * n_b))
      // K_1 is the translation weight
      // K_2 is the rotation weight
      // cm_a and cm_b are the centers of mass for the pointclouds a and b
      // respectively.
      // n_a and n_b are the unit normals for the colinear lines.
      // Start by calculating the center of mass of both pointclouds.
      Vector2T line_a_normal =
        Eigen::Hyperplane<float, 2>::Through(line_a.start,
                                             line_a.endpoint).cast<T>()
                                               .normal();
      Vector2T line_b_normal =
        Eigen::Hyperplane<float, 2>::Through(line_b.start,
                                             line_b.endpoint).cast<T>()
                                               .normal();
      Vector2T a_center_of_mass = a_to_world * center_of_a.cast<T>();
      Vector2T b_center_of_mass = b_to_world * center_of_b.cast<T>();
      Vector2T mass_diff = b_center_of_mass - a_center_of_mass;
      CHECK(ceres::IsFinite(mass_diff.x()));
      CHECK(ceres::IsFinite(mass_diff.y()));
      T translation_error = abs(mass_diff.dot(line_a_normal));
      CHECK(ceres::IsFinite(translation_error));
      T trans_part = T(translation_weight) * translation_error;
      T normal_diff = line_a_normal.dot(line_b_normal);
      CHECK(ceres::IsFinite(normal_diff));
      T rot_part = T(rotation_weight) * (T(1.0) - normal_diff);
      CHECK(ceres::IsFinite(trans_part));
      CHECK(ceres::IsFinite(rot_part));
      residuals[2] = trans_part + rot_part;
      return true;
    }

    ColinearResidual(const LineSegment<float>& line_a,
                     const LineSegment<float>& line_b,
                     const vector<Vector2f>& a_pose_pointcloud,
                     const vector<Vector2f>& b_pose_pointcloud,
                     double translation_weight,
                     double rotation_weight,
                     const LCPose& a_pose,
                     const LCPose& b_pose) :
            line_a(line_a),
            line_b(line_b),
            center_of_a(GetCenterOfMass(a_pose_pointcloud)),
            center_of_b(GetCenterOfMass(b_pose_pointcloud)),
            translation_weight(translation_weight),
            rotation_weight(rotation_weight),
            pose_a_constraint(a_pose),
            pose_b_constraint(b_pose) {}

    static AutoDiffCostFunction<ColinearResidual, 3, 3, 3>*
    create(const LineSegment<float>& line_a,
           const LineSegment<float>& line_b,
           const vector<Vector2f>& a_pose_pointcloud,
           const vector<Vector2f>& b_pose_pointcloud,
           double translation_weight,
           double rotation_weight,
           const LCPose& a_pose,
           const LCPose& b_pose) {
      ColinearResidual* residual = new ColinearResidual(line_a,
                                                        line_b,
                                                        a_pose_pointcloud,
                                                        b_pose_pointcloud,
                                                        translation_weight,
                                                        rotation_weight,
                                                        a_pose,
                                                        b_pose);
      return new AutoDiffCostFunction<ColinearResidual, 3, 3, 3>(residual);
    }

    const LineSegment<float>& line_a;
    const LineSegment<float>& line_b;
    Vector2f center_of_a;
    Vector2f center_of_b;
    double translation_weight;
    double rotation_weight;
    const LCPose pose_a_constraint;
    const LCPose pose_b_constraint;
};

vector<LineSegment<float>> LineSegmentsFromHitlMsg(const HitlSlamInputMsg& msg) {
  Vector2f start_a(msg.line_a_start.x, msg.line_a_start.y);
  Vector2f end_a(msg.line_a_end.x, msg.line_a_end.y);
  Vector2f start_b(msg.line_b_start.x, msg.line_b_start.y);
  Vector2f end_b(msg.line_b_end.x, msg.line_b_end.y);
  vector<LineSegment<float>> lines;
  lines.emplace_back(start_a, end_a);
  lines.emplace_back(start_b, end_b);
  return lines;
}

LCConstraint
GetRelevantPosesForHITL(const HitlSlamInputMsg& hitl_msg,
                        SLAMProblem2D& problem) {
  // Linearly go through all poses
  // Go through all points and see if they lie on either of the two lines.
  const vector<LineSegment<float>> lines = LineSegmentsFromHitlMsg(hitl_msg);
  LCConstraint hitl_constraint(lines[0], lines[1]);
  for (const SLAMNode2D node : problem.nodes) {
    vector<Vector2f> points_on_a;
    vector<Vector2f> points_on_b;
    for (Vector2f point : node.lidar_factor.pointcloud) {
      if (DistanceToLineSegment(point, lines[0]) < HITL_LINE_WIDTH) {
        points_on_a.push_back(point);
      } else if (DistanceToLineSegment(point, lines[0]) < HITL_LINE_WIDTH) {
        points_on_b.push_back(point);
      }
    }
    if (points_on_a.size() >= HITL_POSE_POINT_THRESHOLD) {
      hitl_constraint.line_a_poses.emplace_back(node.node_idx, points_on_a);
    } else if (points_on_b.size() >= HITL_POSE_POINT_THRESHOLD) {
      hitl_constraint.line_b_poses.emplace_back(node.node_idx, points_on_b);
    }
  }
  return hitl_constraint;
}

void Solver::AddColinearConstraints(LCConstraint& constraint) {
  if (constraint.line_a_poses.size() == 0 ||
      constraint.line_b_poses.size() == 0) {
    return;
  }
  loop_closure_constraints_.push_back(constraint);
}

void Solver::AddColinearResiduals(ceres::Problem* problem) {
  for (const LCConstraint constraint : loop_closure_constraints_) {
    for (const LCPose a_pose : constraint.line_a_poses) {
      for (const LCPose b_pose : constraint.line_b_poses) {
        CHECK_LT(a_pose.node_idx, solution_.size());
        CHECK_LT(b_pose.node_idx, solution_.size());
        CHECK_LT(a_pose.node_idx, problem_.nodes.size());
        CHECK_LT(b_pose.node_idx, problem_.nodes.size());
        vector<Vector2f>& pointcloud_a =
          problem_.nodes[a_pose.node_idx].lidar_factor.pointcloud;
        vector<Vector2f>& pointcloud_b =
          problem_.nodes[b_pose.node_idx].lidar_factor.pointcloud;
        problem->
          AddResidualBlock(ColinearResidual::create(constraint.line_a,
                                                    constraint.line_b,
                                                    pointcloud_a,
                                                    pointcloud_b,
                                                    lc_translation_weight_,
                                                    lc_rotation_weight_,
                                                    a_pose,
                                                    b_pose),
                           NULL,
                           solution_[a_pose.node_idx].pose,
                           solution_[b_pose.node_idx].pose);
      }
    }
  }
}

void Solver::SolveForLC() {
  // Create a new problem with colinear residuals between these nodes and a
  // small weighted odometry residuals between all poses and between these ones.
  ceres::Problem problem;
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = std::thread::hardware_concurrency();
  VisualizationCallback vis_callback(problem_, &solution_, n_);
  options.callbacks.push_back(&vis_callback);
  AddOdomFactors(&problem,
                 lc_translation_weight_,
                 lc_rotation_weight_);
  AddColinearResiduals(&problem);
  ceres::Solve(options, &problem, &summary);
}

void Solver::HitlCallback(const HitlSlamInputMsgConstPtr& hitl_ptr) {
  const HitlSlamInputMsg hitl_msg = *hitl_ptr;
  // Get the poses that belong to this input.
  LCConstraint colinear_constraint =
    GetRelevantPosesForHITL(hitl_msg, problem_);
  LOG(INFO) << "Found " << colinear_constraint.line_a_poses.size()
    << " poses for the first line." << std::endl;
  LOG(INFO) << "Found " << colinear_constraint.line_b_poses.size()
            << " poses for the second line." << std::endl;
  AddColinearConstraints(colinear_constraint);
//  SolveForLC();
  // Resolve the initial problem with extra pointcloud residuals between these
  // loop closed points.
  SolveSLAM();
}


//TODO: Take lines from user and find the poses that contain more than a threshold of the selected points.
// Make a HITL residual for colinear poses between each of the pairs.
// Make the odometry correspondences between these poses.
// Solve
// Resolve using lidar_point correspondences and odometry factors between these poses.
