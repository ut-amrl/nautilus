//
// Created by jack on 9/25/19.
//

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "ros/package.h"
#include "sensor_msgs/PointCloud2.h"

#include "solver.h"

#include <utility>
#include "valgrind/memcheck.h"
#include "slam_types.h"
#include "math_util.h"
#include "pointcloud_helpers.h"

using std::vector;
using slam_types::OdometryFactor2D;
using slam_types::LidarFactor;
using ceres::AutoDiffCostFunction;
using Eigen::Matrix2f;
using Eigen::Vector2f;
using slam_types::SLAMNodeSolution2D;

template<typename T> Eigen::Transform<T, 2, Eigen::Affine>
PoseArrayToAffine(const T* rotation, const T* translation) {
  typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
  typedef Eigen::Rotation2D<T> Rotation2DT;
  typedef Eigen::Translation<T, 2> Translation2T;
  Affine2T affine = Rotation2DT(rotation[0]) * Translation2T(translation[0], translation[1]);
  return affine;
}

struct OdometryResidual {
    template <typename T>
    bool operator() (const T* pose_i,
                     const T* pose_j,
                     T* residual) const {
      // Predicted pose_j = pose_i * odometry.
      // Hence, error = pose_j.inverse() * pose_i * odometry;
      // Also A.inverse() = A.transpose() for rotation matrices because they are
      // orthogonal.
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      typedef Eigen::Matrix<T, 2, 2> Matrix2T;

      const Matrix2T Ri = Eigen::Rotation2D<T>(pose_i[2])
              .toRotationMatrix();
      const Matrix2T Rj = Eigen::Rotation2D<T>(pose_j[2])
              .toRotationMatrix();

      const Vector2T Ti(pose_i[0], pose_i[1]);
      const Vector2T Tj(pose_j[0], pose_j[1]);

      const Vector2T error_translation =
              Rj.transpose() * (Ri * T_odom.cast<T>() - (Tj - Ti));

      const Matrix2T error_rotation_mat =
              Rj.transpose() * Ri * R_odom.cast<T>();
      residual[0] = error_translation.x();
      residual[1] = error_translation.y();
      residual[2] = Eigen::Rotation2D<T>().fromRotationMatrix(error_rotation_mat).angle();
      return true;
    }

    explicit OdometryResidual(const OdometryFactor2D& factor) :
            R_odom(Eigen::Rotation2D<float>(factor.rotation)
                    .toRotationMatrix()),
            T_odom(factor.translation) {}

    static AutoDiffCostFunction<OdometryResidual, 3, 3, 3>* create(
            const OdometryFactor2D& factor) {
      OdometryResidual* residual = new OdometryResidual(factor);
      return new AutoDiffCostFunction<OdometryResidual, 3, 3, 3>(residual);
    }

    const Matrix2f R_odom;
    const Vector2f T_odom;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

struct LIDARPointResidual {
    template <typename T>
    bool operator() (const T* source_pose,
                     const T* target_pose,
                     T* residuals) const {
      typedef Eigen::Transform<T, 2, Eigen::Affine> Affine2T;
      typedef Eigen::Matrix<T, 2, 1> Vector2T;
      Affine2T source_to_world = PoseArrayToAffine(&source_pose[2], &source_pose[0]);
      Affine2T target_to_world = PoseArrayToAffine(&target_pose[2], &target_pose[0]);
      Vector2T source_pointT = source_point.cast<T>();
      CHECK(ceres::IsFinite(source_point.x()) && ceres::IsFinite(source_point.y()));
      CHECK(ceres::IsFinite(source_pose[0]) && ceres::IsFinite(source_pose[1]) && ceres::IsFinite(source_pose[2]));
      CHECK(ceres::IsFinite(target_pose[0]) && ceres::IsFinite(target_pose[1]) && ceres::IsFinite(target_pose[2]));
      for (int64_t row = 0; row < source_to_world.rows(); row++) {
        for (int64_t col = 0; col < source_to_world.cols(); col++) {
          CHECK(ceres::IsFinite(source_to_world(row, col)));
          CHECK(ceres::IsFinite(target_to_world.inverse()(row, col)));
        }
      }
      Vector2T target_pointT = target_point.cast<T>();
      // Transform source_point into the frame of target_point
      CHECK(ceres::IsFinite(source_pointT.x()) && ceres::IsFinite(source_pointT.y()));
      source_pointT = target_to_world.inverse() * source_to_world * source_pointT;
//      CHECK(ceres::IsFinite(source_pointT.x()) && ceres::IsFinite(source_pointT.y()));
      residuals[0] = source_pointT.x() - target_pointT.x();
      residuals[1] = source_pointT.y() - target_pointT.y();
      return true;
    }

    LIDARPointResidual(Vector2f& source_point,
                       Vector2f& target_point) :
            source_point(source_point),
            target_point(target_point) {}

    static AutoDiffCostFunction<LIDARPointResidual, 2, 3, 3>* create(
            Vector2f& source_point, Vector2f& target_point) {
      LIDARPointResidual *residual = new LIDARPointResidual(source_point, target_point);
      return new AutoDiffCostFunction<LIDARPointResidual, 2, 3, 3>(residual);
    }

    const Vector2f source_point;
    const Vector2f target_point;
};

class VisualizationCallback : public ceres::IterationCallback {
public:
    VisualizationCallback(const slam_types::SLAMProblem2D& problem,
            const vector<slam_types::SLAMNodeSolution2D>* solution,
            ros::NodeHandle& n) : problem(problem), solution(solution) {
      pointcloud_helpers::InitPointcloud(&all_points_marker);
      all_points.clear();
      point_pub = n.advertise<sensor_msgs::PointCloud2>("/all_points", 10);
    }

    void PubVisualization() {
      const vector<slam_types::SLAMNodeSolution2D>& solution_c = *solution;
      for (size_t i = 0; i < solution_c.size(); i++) {
        auto pointcloud = problem.nodes[i].lidar_factor.pointcloud;
        (void) pointcloud;
        Eigen::Affine2f robot_to_world = PoseArrayToAffine(&(solution_c[i].pose[2]), &(solution_c[i].pose[0])).cast<float>();
        for (const Vector2f& point : pointcloud) {
          all_points.push_back(robot_to_world * point);
        }
      }
      pointcloud_helpers::PublishPointcloud(all_points, all_points_marker, point_pub);
      ros::spinOnce();
      all_points.clear();
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
      PubVisualization();
      return ceres::SOLVER_CONTINUE;
    }

private:
    sensor_msgs::PointCloud2 all_points_marker;
    std::vector<Vector2f> all_points;
    const slam_types::SLAMProblem2D& problem;
    const vector<slam_types::SLAMNodeSolution2D>* solution;
    ros::Publisher point_pub;
};

void AddOdomFactors(const slam_types::SLAMProblem2D& problem,
                    vector<slam_types::SLAMNodeSolution2D>& solution,
                    ceres::Problem* ceres_problem, size_t node_i_index, size_t node_j_index) {

  for (const OdometryFactor2D& odom_factor : problem.odometry_factors) {
    CHECK_LT(odom_factor.pose_i, odom_factor.pose_j);
    CHECK_GT(solution.size(), odom_factor.pose_i);
    CHECK_GT(solution.size(), odom_factor.pose_j);
    ceres_problem->AddResidualBlock(OdometryResidual::create(odom_factor),
                                    NULL,
                                    solution[odom_factor.pose_i].pose,
                                    solution[odom_factor.pose_j].pose);
  }
}


#define LIDAR_CONSTRAINT_AMOUNT 10

// Source moves to target.
double AddPointCloudFactors(const slam_types::SLAMProblem2D& problem,
                            vector<slam_types::SLAMNodeSolution2D>* solution_ptr,
                            ceres::Problem* ceres_problem,
                            size_t source_node_index,
                            size_t target_node_index) {
  // Loop over the two point clouds finding the closest points.
  double difference = 0.0;
  vector<SLAMNodeSolution2D>& solution = *solution_ptr;
  SLAMNodeSolution2D& source_solution = solution[source_node_index];
  SLAMNodeSolution2D& target_solution = solution[target_node_index];
  const vector<Vector2f>& source_pointcloud =
          problem.nodes[source_node_index].lidar_factor.pointcloud;
  const vector<Vector2f>& target_pointcloud =
          problem.nodes[target_node_index].lidar_factor.pointcloud;
  Eigen::Affine2f source_to_world =
          PoseArrayToAffine(&source_solution.pose[2], &source_solution.pose[0]).cast<float>();
  Eigen::Affine2f target_to_world =
          PoseArrayToAffine(&target_solution.pose[2], &target_solution.pose[0]).cast<float>();
  for (const Vector2f& source_point : source_pointcloud) {
    Vector2f source_point_transformed =
            target_to_world.inverse() * source_to_world * source_point;
    Vector2f closest_target = target_pointcloud[0];
    float min_distance = (closest_target - source_point_transformed).norm();
    for(const Vector2f& target_point : target_pointcloud) {
      if ((target_point - source_point_transformed).norm() < min_distance) {
        closest_target = target_point;
        min_distance = (target_point - source_point_transformed).norm();
      }
    }
    difference += (closest_target - source_point_transformed).norm();
    // Minimize distance between closest points!
    Vector2f source_point_modifiable = source_point;
    ceres_problem->AddResidualBlock(LIDARPointResidual::create(source_point_modifiable, closest_target),
            NULL,
            source_solution.pose,
            target_solution.pose);
  }
  return difference;
}

bool solver::SolveSLAM(slam_types::SLAMProblem2D& problem, ros::NodeHandle& n) {
  // Copy all the data to a list that we are going to modify as we optimize.
  vector<slam_types::SLAMNodeSolution2D> solution(problem.nodes.size());
  for (size_t i = 0; i < problem.nodes.size(); i++) {
    // Make sure that we marked all the data correctly earlier.
    CHECK_EQ(i, problem.nodes[i].node_idx);
    SLAMNodeSolution2D sol_node(problem.nodes[i]);
    solution[i] = sol_node;
  }
  // Setup ceres for evaluation of the problem.
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  // Add the visualization.
  VisualizationCallback vis_callback(problem, &solution, n);
  options.callbacks.push_back(&vis_callback);
  // Continually solve and minimize.
  for (size_t node_i_index = LIDAR_CONSTRAINT_AMOUNT;
       node_i_index < problem.nodes.size();
       node_i_index++) {
    double difference = 10;
    do {
      for (size_t node_j_index = node_i_index - LIDAR_CONSTRAINT_AMOUNT;
           node_j_index < node_i_index;
           node_j_index++) {
        ceres::Problem ceres_problem;
        // Add all the points to this, make a new problem. Minimize, continue.
        AddOdomFactors(problem, solution, &ceres_problem, node_i_index,
                       node_j_index);
        difference = AddPointCloudFactors(problem,
                                          &solution,
                                          &ceres_problem,
                                          node_j_index,
                                          node_i_index);
        ceres::Solve(options, &ceres_problem, &summary);
        //printf("%s\n", summary.FullReport().c_str());
      }
    } while(difference > LIDAR_CONSTRAINT_AMOUNT * 0.1);
  }
  return true;
}

