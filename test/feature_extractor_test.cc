#include "sensor_msgs/LaserScan.h"

#include "gflags/gflags.h"

#include "config_reader/config_reader.h"
#include "../src/util/slam_types.h"
#include "../src/visualization/solver_vis_ros.h"
#include "../src/optimization/solver.h"

using Eigen::Vector2f;
using std::vector;
using namespace nautilus;
using namespace slam_types;

CONFIG_STRING(bag_path, "bag_path");
CONFIG_BOOL(auto_lc, "auto_lc");
CONFIG_STRING(lc_debug_output_dir, "lc_debug_output_dir");
CONFIG_STRING(lidar_topic, "lidar_topic");
CONFIG_STRING(odom_topic, "odom_topic");
CONFIG_STRING(hitl_lc_topic, "hitl_lc_topic");
CONFIG_BOOL(differential_odom, "differential_odom");
DEFINE_string(config_file, "", "The path to the config file to use.");

int main(int argc, char ** argv) {	
  google::InitGoogleLogging(*argv);
  google::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_config_file == "") {
    printf("Must specify a config file!\n");
    exit(1);
  }
  config_reader::ConfigReader reader({FLAGS_config_file});
	ros::init(argc, argv, "nautilus");
	ros::NodeHandle n;
  ros::service::waitForService("/rviz/get_loggers");
  sleep(5);
  std::cout << "Building Corner" << std::endl;
	vector<Vector2f> corner;
	for (float i = 0.5f; i >= 0.0f; i -= 0.02f) {
		corner.emplace_back(i, 0.0f);
	}
  for (float i = 0.02f; i < 0.5f; i += 0.02f) {
    corner.emplace_back(0.0f, i);
  }
	// Construct the problem from this.
	// We are going to be assuming they are 25 cm apart.
  std::cout << "Building Slam Types" << std::endl;
	float new_x = -0.15f;
	sensor_msgs::LaserScan laser_scan;
  std::cout << "\tLidar Factors" << std::endl;
	LidarFactor lf1(0, laser_scan, corner);
	LidarFactor lf2(1, laser_scan, corner);
  std::cout << "\tPoses" << std::endl;
	RobotPose2D pose1(Vector2f(0.0f, 0.0f), 0.0f);
	RobotPose2D pose2(Vector2f(new_x, 0.0f), 0.2f);
	Vector2f translation(new_x, 0.0f);
  std::cout << "\tOdom" << std::endl;
	OdometryFactor2D odom1(0, 1, translation, 0.2f);
	vector<SLAMNode2D> nodes;
	nodes.emplace_back(0, 0.0f, pose1, lf1);
	nodes.emplace_back(1, 0.0f, pose2, lf2);
	vector<OdometryFactor2D> odom_factors = {odom1};
	SLAMProblem2D problem(nodes, odom_factors);
  std::cout << "Building State" << std::endl;
	std::shared_ptr<SLAMState2D> state = std::make_shared<SLAMState2D>(problem);
	std::unique_ptr<nautilus::visualization::SolverVisualizerROS> vis =
      std::make_unique<nautilus::visualization::SolverVisualizerROS>(state, n);
  std::cout << "Solving" << std::endl;
  Solver solver(n, state, std::move(vis));
  solver.SolveSLAM();
  std::cout << "Finished" << std::endl;

  while (ros::ok()) {}
}
