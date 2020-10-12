#include <csignal>
#include <vector>

#include "input/slam_type_builder.h"
#include "util/slam_types.h"
#include "optimization/solver.h"
#include "config_reader/config_reader.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "nautilus/CobotOdometryMsg.h"
#include "nav_msgs/Odometry.h"
#include "ros/node_handle.h"
#include "ros/package.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/LaserScan.h"

namespace nautilus {

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
DEFINE_string(solution_poses, "", "The path to the file containing the solution poses, to load from an existing solution.");

static bool asking_for_input = true;

SLAMProblem2D ProcessBagFile(const char *bag_path, const ros::NodeHandle &n) {
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
  } catch (rosbag::BagException &exception) {
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
      printf("Processed %ld messages of %d, found %ld nodes.\n", msg_counter, view.size(), slam_builder.GetNodeCount());
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
  printf("Exiting with %d\n", signum);
  asking_for_input = false;
  ros::shutdown();
  exit(0);
}

void PopulateSLAMProblem(SLAMProblem2D& slam_problem, Solver& solver) {
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
}

void SolveSLAMProblem(SLAMProblem2D& slam_problem, Solver& solver) {
  std::cout << "Solving SLAM Problem..." << std::endl;
  solver.SolveSLAM();
  // Do a final pass through and check for any LC nodes.
  // But only if automatic loop closure is enabled.
  if (CONFIG_auto_lc) {
    std::cout << "Automatically loop closing" << std::endl;
    std::ofstream lc_output_file;
    lc_output_file.open(CONFIG_lc_debug_output_dir + "/lc_matches.txt",
                        std::ios::trunc);
    lc_output_file.close();
    for (SLAMNode2D &node : slam_problem.nodes) {
      solver.CheckForLearnedLC(node);
    }
    solver.SolveSLAM();
  }
}

}  // namespace nautilus

using nautilus::Solver;

int main(int argc, char **argv) {
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
  ros::init(argc, argv, "nautilus");
  ros::NodeHandle n;
  signal(SIGINT, nautilus::SignalHandler);
  // Load and pre-process the data.
  std::string package_path = ros::package::getPath("nautilus");
  nautilus::slam_types::SLAMProblem2D slam_problem = nautilus::ProcessBagFile(
      (package_path + "/" + nautilus::CONFIG_bag_path).c_str(), n);
  CHECK_GT(slam_problem.nodes.size(), 1)
      << " Not enough nodes were processed"
      << " you probably didn't specify the correct topics!\n";
  // Load all the residuals into the problem and run to get initial solution.
  Solver solver(n);
  nautilus::PopulateSLAMProblem(slam_problem, solver);
  if (nautilus::FLAGS_solution_poses != "") {
    std::cout << "Loading solution poses; skipping SLAM solving step." << std::endl;
    solver.LoadSLAMSolution(nautilus::FLAGS_solution_poses.c_str());
  } else {
    nautilus::SolveSLAMProblem(slam_problem, solver);
  }

  std::cout << "Waiting for Loop Closure input" << std::endl;
  ros::Subscriber hitl_sub =
      n.subscribe(nautilus::CONFIG_hitl_lc_topic, 10, &Solver::HitlCallback, &solver);
  ros::Subscriber write_sub =
      n.subscribe("/write_output", 10, &Solver::WriteCallback, &solver);
  ros::Subscriber vector_sub =
      n.subscribe("/vectorize_output", 10, &Solver::Vectorize, &solver);
  ros::spin();
  return 0;
}
