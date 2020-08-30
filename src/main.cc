#include <csignal>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "nav_msgs/Odometry.h"
#include "ros/node_handle.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/LaserScan.h"

#include "./slam_type_builder.h"
#include "./slam_types.h"
#include "./solver.h"
#include "config_reader/config_reader.h"
#include "nautilus/CobotOdometryMsg.h"

using nautilus::CobotOdometryMsg;
using nautilus::CobotOdometryMsgPtr;
using nautilus::HitlSlamInputMsg;
using nautilus::WriteMsg;
using slam_types::SLAMNode2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
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
DEFINE_string(solution_poses,
              "",
              "The path to the file containing the solution poses, to load "
              "from an existing solution.");

bool TryProcessLaser(const rosbag::MessageInstance& message,
                     SLAMTypeBuilder* slam_builder) {
  auto laser_scan = message.instantiate<sensor_msgs::LaserScan>();
  if (laser_scan == nullptr) {
    return false;
  }
  slam_builder->LidarCallback(*laser_scan);
  return true;
}

bool TryProcessOdom(const rosbag::MessageInstance& message,
                    SLAMTypeBuilder* slam_builder) {
  auto std_odom = message.instantiate<nav_msgs::Odometry>();
  if (std_odom != nullptr) {
    slam_builder->OdometryCallback(*std_odom);
    return true;
  }
  auto cobot_odom = message.instantiate<CobotOdometryMsg>();
  if (cobot_odom != nullptr) {
    if (!CONFIG_differential_odom) {
      std::cout << "Error: Recieved Cobot odometry message, but "
                   "differential odometry is not enabled."
                << std::endl;
      exit(1);
    }
    slam_builder->OdometryCallback(*cobot_odom);
    return true;
  }
  return false;
}

SLAMProblem2D ProcessBagFile(const std::string& bag_path,
                             const ros::NodeHandle& n) {
  /*
   * Loads and processes the bag file pulling out the lidar data
   * and the odometry data. Keeps track of the current pose and produces
   * a list of poses / pointclouds. Also keeps a list of the odometry data.
   */
  std::cout << "Loading bag file..." << std::endl;
  rosbag::Bag bag;
  try {
    bag.open(bag_path, rosbag::bagmode::Read);
  } catch (const rosbag::BagException& exception) {
    std::cout << "Unable to read " << bag_path
              << ", reason: " << exception.what() << std::endl;
    return {};
  }
  bool found_odom = false;
  bool found_lidar = false;
  std::cout << "Laser topic: " << CONFIG_lidar_topic << std::endl;
  std::cout << "Odom topic: " << CONFIG_odom_topic << std::endl;
  SLAMTypeBuilder slam_builder;
  size_t msg_counter = 0;
  // Iterate through the bag
  rosbag::View view(
      bag, rosbag::TopicQuery({CONFIG_lidar_topic, CONFIG_odom_topic}));
  for (const auto& msg : view) {
    if (slam_builder.Done() || !ros::ok()) {
      break;
    }
    msg_counter++;
    found_lidar |= TryProcessLaser(msg, &slam_builder);
    found_odom |= TryProcessOdom(msg, &slam_builder);

    if (msg_counter % 5000 == 0) {
      std::cout << "Processed " << msg_counter << " messages of " << view.size()
                << ", found " << slam_builder.GetNodeCount() << " nodes.\n";
    }
  }
  bag.close();

  if (!found_lidar) {
    std::cout << "Did not find any lidar scans! Please check your specified "
                 "topics.\n";
    return {};
  }
  std::cout << "Successfully found lidar messages.\n";
  if (!found_odom) {
    std::cout << "Did not find any odometry messages! Please check your "
                 "specified topics.\n";
    return {};
  }
  std::cout << "Successfully found odometry messages.\n";

  std::cout << "Done." << std::endl;
  return slam_builder.GetSlamProblem();
}

void SignalHandler(int signum) {
  std::cout << "Exiting with SIGNAL: " << strsignal(signum) << std::endl;
  ros::shutdown();
  exit(0);
}

void PopulateSLAMProblem(SLAMProblem2D* slam_problem, Solver* solver) {
  // Iteratively add all the nodes and odometry factors.
  CHECK_EQ(slam_problem->nodes.size(),
           slam_problem->odometry_factors.size() + 1);
  size_t node_index = 0;
  for (; node_index < slam_problem->nodes.size(); node_index++) {
    if (node_index == 0) {
      solver->AddSlamNode(slam_problem->nodes[0]);
      continue;
    }
    solver->AddSLAMNodeOdom(slam_problem->nodes[node_index],
                            slam_problem->odometry_factors[node_index - 1]);
  }
  std::cout << "Nodes added: " << node_index + 1 << std::endl;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(*argv);
  google::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_config_file == "") {
    std::cout << "Must specify a config file!\n";
    return 1;
  }
  config_reader::ConfigReader reader({FLAGS_config_file});
  if (CONFIG_bag_path == "") {
    std::cout << "Must specify an input bag!\n";
    return 1;
  }
  ros::init(argc, argv, "nautilus");
  ros::NodeHandle n;
  signal(SIGINT, SignalHandler);
  // Load and pre-process the data.
  SLAMProblem2D slam_problem = ProcessBagFile(CONFIG_bag_path, n);
  CHECK_GT(slam_problem.nodes.size(), 1)
      << " Not enough nodes were processed, you probably didn't specify the "
         "correct topics!\n";
  // Load all the residuals into the problem and run to get initial solution.
  Solver solver(n);
  PopulateSLAMProblem(&slam_problem, &solver);
  if (FLAGS_solution_poses != "") {
    std::cout << "Loading solution poses; skipping SLAM solving step."
              << std::endl;
    solver.LoadSLAMSolution(FLAGS_solution_poses);
  } else {
    std::cout << "Performing initial solve\n";
    solver.SolveSLAM();
  }

  std::cout << "Waiting for Loop Closure input" << std::endl;
  ros::Subscriber hitl_sub =
      n.subscribe(CONFIG_hitl_lc_topic, 10, &Solver::HitlCallback, &solver);
  ros::Subscriber write_sub =
      n.subscribe("/write_output", 10, &Solver::WriteCallback, &solver);
  ros::Subscriber vector_sub =
      n.subscribe("/vectorize_output", 10, &Solver::Vectorize, &solver);
  ros::spin();
  return 0;
}
