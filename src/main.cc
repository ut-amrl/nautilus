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
#include "lidar_slam/CobotOdometryMsg.h"

using lidar_slam::CobotOdometryMsg;
using lidar_slam::HitlSlamInputMsg;
using lidar_slam::WriteMsg;
using slam_types::SLAMNode2D;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using std::string;
using std::vector;

CONFIG_STRING(bag_path, "bag_path");
CONFIG_BOOL(auto_lc, "auto_lc");
CONFIG_STRING(lidar_topic, "lidar_topic");
CONFIG_STRING(odom_topic, "odom_topic");
CONFIG_STRING(hitl_lc_topic, "hitl_lc_topic");

DEFINE_string(config_file, "", "The path to the config file to use.");

static bool asking_for_input = true;

SLAMProblem2D ProcessBagFile(const char* bag_path, const ros::NodeHandle& n) {
  /*
   * Loads and processes the bag file pulling out the lidar data
   * and the odometry data. Keeps track of the current pose and produces
   * a list of poses / pointclouds. Also keeps a list of the odometry data.
   */
  printf("Loading bag file... ");
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
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  SLAMTypeBuilder slam_builder;
  // Iterate through the bag
  for (rosbag::View::iterator it = view.begin();
       ros::ok() && it != view.end() && !slam_builder.Done(); ++it) {
    const rosbag::MessageInstance& message = *it;
    {
      // Load all the point clouds into memory.
      sensor_msgs::LaserScanPtr laser_scan =
          message.instantiate<sensor_msgs::LaserScan>();
      if (laser_scan != nullptr) {
        slam_builder.LidarCallback(*laser_scan);
      }
    }
    {
      nav_msgs::OdometryPtr odom = message.instantiate<nav_msgs::Odometry>();
      if (odom != nullptr) {
        slam_builder.OdometryCallback(*odom);
      }
    }
    {
      lidar_slam::CobotOdometryMsgPtr odom =
          message.instantiate<CobotOdometryMsg>();
      if (odom != nullptr) {
        slam_builder.OdometryCallback(*odom);
      }
    }
  }
  bag.close();
  printf("Done.\n");
  fflush(stdout);
  return slam_builder.GetSlamProblem();
}

void SignalHandler(int signum) {
  printf("Exiting with %d\n", signum);
  asking_for_input = false;
  ros::shutdown();
  exit(0);
}

void LearnedLoopClosure(SLAMProblem2D& slam_problem, Solver& solver) {
  // Iteratively add all the nodes and odometry factors.
  for (uint64_t node_index = 0; node_index < slam_problem.nodes.size();
       node_index++) {
    if (node_index == 0) {
      solver.AddSlamNode(slam_problem.nodes[0]);
    } else {
      solver.AddSLAMNodeOdom(slam_problem.nodes[node_index],
                             slam_problem.odometry_factors[node_index - 1]);
    }
    std::cout << "Nodes added: " << node_index + 1 << std::endl;
  }
  std::cout << "Solving initial" << std::endl;
  solver.SolveSLAM();
  // Do a final pass through and check for any LC nodes.
  // But only if automatic loop closure is enabled.
  if (CONFIG_auto_lc) {
    std::cout << "Automatically loop closing" << std::endl;
    for (SLAMNode2D& node : slam_problem.nodes) {
      solver.CheckForLearnedLC(node);
    }
    solver.SolveSLAM();
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(*argv);
  google::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_config_file.compare("") == 0) {
    printf("Must specify a config file!\n");
    exit(1);
  }
  config_reader::ConfigReader reader({FLAGS_config_file});
  if (CONFIG_bag_path.compare("") == 0) {
    printf("Must specify an input bag!\n");
    exit(1);
  }
  ros::init(argc, argv, "lidar_slam");
  ros::NodeHandle n;
  signal(SIGINT, SignalHandler);
  // Load and pre-process the data.
  SLAMProblem2D slam_problem = ProcessBagFile(CONFIG_bag_path.c_str(), n);
  CHECK_GT(slam_problem.nodes.size(), 1)
      << " Not enough nodes were processed"
      << " you probably didn't specify the correct topics!\n";
  // Load all the residuals into the problem and run to get initial solution.
  Solver solver(n);
  LearnedLoopClosure(slam_problem, solver);
  std::cout << "Waiting for Loop Closure input" << std::endl;
  ros::Subscriber hitl_sub =
      n.subscribe(CONFIG_hitl_lc_topic, 10, &Solver::HitlCallback, &solver);
  ros::Subscriber write_sub =
      n.subscribe("/write_output", 10, &Solver::WriteCallback, &solver);
  ros::Subscriber vector_sub =
      n.subscribe("/vectorize_output", 10, &Solver::Vectorize, &solver);
  ros::spin();
  //  while (asking_for_input) {
  //    uint64_t scan_a, scan_b;
  //    std::cout << "Type a pose A: "
  //    std::cin >> scan_a;
  //    std::cout << std::endl << "Type a pose B: ";
  //    std::cin >> scan_b;
  //    std::cout << std::endl;
  //    // Run ChiSquare on these two scans.
  //    if (scan_a > slam_problem.nodes.size() | scan_b >
  //    slam_problem.nodes.size()) {
  //      continue;
  //    }
  //    int chi_num = solver.GetChiSquare();
  //    chi_squared dist(3);
  //    return boost::math::cdf(dist, chi_num) <= certainty;
  //    std::cout << "ChiSquare Boundary is: " << boost::math::cdf(dist,
  //    chi_num);
  //  }
  return 0;
}
