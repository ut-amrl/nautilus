#include <csignal>
#include <vector>

#include "Eigen/Dense"
#include "config_reader/config_reader.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "input/slam_type_builder.h"
#include "nautilus/CobotOdometryMsg.h"
#include "nav_msgs/Odometry.h"
#include "optimization/solver.h"
#include "ros/node_handle.h"
#include "ros/package.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/LaserScan.h"
#include "util/slam_types.h"
#include "visualization/solver_vis_ros.h"

namespace nautilus {

using nautilus::CobotOdometryMsg;
using nautilus::CobotOdometryMsgPtr;
using nautilus::HitlSlamInputMsg;
using nautilus::WriteMsg;
using slam_types::SLAMNode2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using slam_types::SLAMState2D;
using std::string;
using std::vector;

CONFIG_STRING(bag_path, "bag_path");
CONFIG_BOOL(auto_lc, "auto_lc");
CONFIG_STRING(lc_debug_output_dir, "lc_debug_output_dir");
CONFIG_STRING(lidar_topic, "lidar_topic");
CONFIG_STRING(odom_topic, "odom_topic");
CONFIG_STRING(hitl_lc_topic, "hitl_lc_topic");
CONFIG_BOOL(differential_odom, "differential_odom");

DEFINE_string(config_file, "", "The path to the config file to use.");
DEFINE_string(solution_poses, "",
              "The path to the file containing the solution poses, to load "
              "from an existing solution.");

static bool asking_for_input = true;

SLAMProblem2D ProcessBagFile(const char* bag_path, const ros::NodeHandle& n) {
  /*
   * Loads and processes the bag file pulling out the lidar data
   * and the odometry data. Keeps track of the current pose and produces
   * a list of poses / pointclouds. Also keeps a list of the odometry data.
   */
  printf("Loading bag file [%s] ...\n", bag_path);
  fflush(stdout);
  rosbag::Bag bag;
  try {
    bag.open(bag_path, rosbag::bagmode::Read);
  } catch (rosbag::BagException& exception) {
    printf("Unable to read %s, reason: %s\n", bag_path, exception.what());
    return slam_types::SLAMProblem2D();
  }
  // Get the topics we want
  vector<string> topics;
  topics.emplace_back(CONFIG_odom_topic.c_str());
  topics.emplace_back(CONFIG_lidar_topic.c_str());
  bool found_odom = false;
  bool found_lidar = false;
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  SLAMTypeBuilder slam_builder;
  size_t msg_counter = 0;
  // Iterate through the bag
  for (rosbag::View::iterator it = view.begin();
       ros::ok() && it != view.end() && !slam_builder.Done(); ++it) {
    const rosbag::MessageInstance& message = *it;
    msg_counter++;
    {
      // Load all the point clouds into memory.
      sensor_msgs::LaserScanPtr laser_scan =
          message.instantiate<sensor_msgs::LaserScan>();
      if (laser_scan != nullptr) {
        found_lidar = true;
        slam_builder.LidarCallback(&(*laser_scan));
      }
    }
    {
      nav_msgs::OdometryPtr odom = message.instantiate<nav_msgs::Odometry>();
      if (odom != nullptr) {
        found_odom = true;
        slam_builder.OdometryCallback(*odom);
      }
    }
    {
      CobotOdometryMsgPtr odom = message.instantiate<CobotOdometryMsg>();
      if (odom != nullptr) {
        if (!CONFIG_differential_odom) {
          printf(
              "Error: Recieved Cobot odometry message, but differential "
              "odometry is not enabled.\n");
          exit(1);
        }
        found_odom = true;
        slam_builder.OdometryCallback(*odom);
      }
    }

    if (msg_counter % 5000 == 0) {
      printf("Processed %ld messages of %d, found %ld nodes.\n", msg_counter,
             view.size(), slam_builder.GetNodeCount());
    }
  }

  if (!found_lidar) {
    printf(
        "Did not find any lidar scans! Please check your specified topics.\n");
  } else {
    printf("Successfully found lidar messages.\n");
  }
  if (!found_odom) {
    printf(
        "Did not find any odometry messages! Please check your specified "
        "topics.\n");
  } else {
    printf("Successfully found odometry messages.\n");
  }

  bag.close();
  printf("Done.\n");
  fflush(stdout);
  return slam_builder.GetSlamProblem();
}

void SignalHandler(int signum) {
  asking_for_input = false;
  std::cout << "Shutting down" << std::endl;
  ros::shutdown();
  std::cout << "ROS Shutdown" << std::endl;
  exit(0);
}

void LoadSolutionFromFile(std::shared_ptr<SLAMState2D> state, std::string poses_path) {
  std::map<double, Eigen::Vector3f> poses;
  std::ifstream poses_file;
  poses_file.open(poses_path);
  if (poses_file.is_open()) {
    double timestamp;
    float pose_x, pose_y, theta;
    while (poses_file >> timestamp >> pose_x >> pose_y >> theta) {
      poses[timestamp] = Eigen::Vector3f(pose_x, pose_y, theta);
    }
  }
  poses_file.close();
  std::cout << "Finished loading solution file." << std::endl;
  for (size_t i = 0; i < state->solution.size(); i++) {
    std::stringstream ss;
    ss << std::fixed << state->solution[i].timestamp;
    double timestamp = std::stod(ss.str());
    if (poses.find(timestamp) != poses.end()) {
      state->solution[i].pose[0] = poses[timestamp][0];
      state->solution[i].pose[1] = poses[timestamp][1];
      state->solution[i].pose[2] = poses[timestamp][2];
    } else {
      printf("Unable to find solution for timestamp %f\n", timestamp);
    }
  }
}

}  // namespace nautilus

using nautilus::Solver;
using nautilus::slam_types::SLAMState2D;

int main(int argc, char** argv) {
  google::InitGoogleLogging(*argv);
  google::ParseCommandLineFlags(&argc, &argv, false);
  if (nautilus::FLAGS_config_file == "") {
    printf("Must specify a config file!\n");
    exit(1);
  }
  config_reader::ConfigReader reader({nautilus::FLAGS_config_file});
  if (nautilus::CONFIG_bag_path == "") {
    printf("Must specify an input bag!\n");
    exit(1);
  }
  ros::init(argc, argv, "nautilus", ros::init_options::NoSigintHandler);
  ros::NodeHandle n;
  signal(SIGINT, nautilus::SignalHandler);
  // Load and pre-process the data.
  std::string package_path = ros::package::getPath("nautilus");
  nautilus::slam_types::SLAMProblem2D slam_problem = nautilus::ProcessBagFile(
      (package_path + "/" + nautilus::CONFIG_bag_path).c_str(), n);
  CHECK_GT(slam_problem.nodes.size(), 1)
      << " Not enough nodes were processed"
      << " you probably didn't specify the correct topics!\n";
  // Construct the slam state.
  std::shared_ptr<SLAMState2D> state = std::make_shared<SLAMState2D>(slam_problem);
  // Check if there is a pre-existing solution file to load.
  if (nautilus::FLAGS_solution_poses != "") {
    std::cout << "Loading solution poses; skipping SLAM solving step."
              << std::endl;
    nautilus::LoadSolutionFromFile(state, nautilus::FLAGS_solution_poses);
  }

  // Load all the residuals into the problem and run to get initial solution.
  std::unique_ptr<nautilus::visualization::SolverVisualizerROS> vis =
      std::make_unique<nautilus::visualization::SolverVisualizerROS>(state, n);
  Solver solver(n, state, std::move(vis));
  // Wait for RViz to start before we solve so visualization will be displayed.
  ros::service::waitForService("/rviz/reload_shaders");
  // Call the solver once.
  solver.SolveSLAM();

  // TODO :Remove
//  double last_x = 0.0;
//  double last_y = 0.0;
//  for (const auto& sol_node : state->solution) {
//    Eigen::Vector2f diff(sol_node.pose[0] - last_x, sol_node.pose[1] - last_y);
//    if (diff.norm() >= 0.35) {
//      std::cout << "Difference Between " << sol_node.node_idx << " " << (sol_node.node_idx - 1) << std::endl;
//    }
//    last_x = sol_node.pose[0];
//    last_y = sol_node.pose[1];
//  }

  std::cout << "Waiting for Loop Closure input" << std::endl;
  ros::Subscriber hitl_sub = n.subscribe(nautilus::CONFIG_hitl_lc_topic, 10,
                                         &Solver::HitlCallback, &solver);
  ros::Subscriber write_sub =
      n.subscribe("/write_output", 10, &Solver::WriteCallback, &solver);
  ros::Subscriber vector_sub =
      n.subscribe("/vectorize_output", 10, &Solver::Vectorize, &solver);
  // Wait for input
  while (ros::ok()) {
    ros::spinOnce();
  }
  return 0;
}
