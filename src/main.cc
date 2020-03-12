#include <csignal>
#include <vector>

#include "ros/node_handle.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/Odometry.h"

#include "./slam_type_builder.h"
#include "./slam_types.h"
#include "./solver.h"
#include "lidar_slam/CobotOdometryMsg.h"
#include "lidar_slam/WriteMsg.h"

using std::string;
using std::vector;
using slam_types::SLAMNodeSolution2D;
using slam_types::SLAMProblem2D;
using slam_types::SLAMNode2D;
using lidar_slam::CobotOdometryMsg;
using lidar_slam::HitlSlamInputMsg;
using lidar_slam::WriteMsg;

DEFINE_string(
  bag_path,
  "",
  "The location of the bag file to run SLAM on.");
DEFINE_string(
  odom_topic,
  "/odometry/filtered",
  "The topic that odometry messagse are published over.");
DEFINE_string(
  lidar_topic,
  "/velodyne_2dscan_high_beams",
  "The topic that lidar messages are published over.");
DEFINE_double(
  translation_weight,
  1.0,
  "Weight multiplier for changing the odometry predicted translation.");
DEFINE_double(
  rotation_weight,
  1.0,
  "Weight multiplier for changing the odometry predicted rotation.");
DEFINE_double(
  stopping_accuracy,
  0.05,
  "Threshold of accuracy for stopping.");
DEFINE_int64(
  pose_num,
  30,
  "The number of poses to process.");
DEFINE_bool(
  diff_odom,
  false,
  "Is the odometry differential (True for CobotOdometryMsgs)?");
DEFINE_double(
  lc_translation_weight,
  2,
  "The translation weight for loop closure.");
DEFINE_double(
  lc_rotation_weight,
  2,
  "The rotational weight for loop closure.");
DEFINE_string(
  hitl_lc_topic,
  "/hitl_slam_input",
  "The topic which the HITL line messages are published over.");
DEFINE_double(
   max_lidar_range,
  0,
  "The user specified range cutoff for lidar range data, if not used will be the sensor default specified in the bag (meters)");
DEFINE_string(
    pose_output_file,
    "poses.txt",
    "The file to output the finalized poses into");


SLAMProblem2D ProcessBagFile(const char* bag_path,
                             const ros::NodeHandle& n) {
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
  } catch (rosbag::BagException &exception) {
    printf("Unable to read %s, reason: %s\n", bag_path, exception.what());
    return slam_types::SLAMProblem2D();
  }
  // Get the topics we want
  vector<string> topics;
  topics.emplace_back(FLAGS_odom_topic.c_str());
  topics.emplace_back(FLAGS_lidar_topic.c_str());
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  SLAMTypeBuilder slam_builder(FLAGS_pose_num,
                               FLAGS_diff_odom,
                               FLAGS_max_lidar_range);
  bool bag_start_time_printed = false;
  // Iterate through the bag
  for (rosbag::View::iterator it = view.begin();
       ros::ok() && it != view.end() && !slam_builder.Done();
       ++it) {
    const rosbag::MessageInstance &message = *it;
    if (!bag_start_time_printed) {
      bag_start_time_printed = true;
      ros::Time current_time = message.getTime();
      std::cout << "Start Time: " << current_time.sec << " s " << current_time.nsec << " n " << std::endl;
    }
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
      lidar_slam::CobotOdometryMsgPtr odom = message.instantiate<CobotOdometryMsg>();
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
  ros::shutdown();
  exit(0);
}

void LearnedLoopClosure(SLAMProblem2D& slam_problem,
                        Solver& solver) {
  // Iteratively add all the nodes and odometry factors.
  for (uint64_t node_index = 0;
       node_index < slam_problem.nodes.size();
       node_index++) {
    if (node_index == 0) {
      solver.AddSlamNode(slam_problem.nodes[0]);
    } else {
      solver.AddSLAMNodeOdom(slam_problem.nodes[node_index],
                             slam_problem.odometry_factors[node_index - 1]);
    }
  }
  solver.SolveSLAM();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(*argv);
  google::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_bag_path.compare("") == 0) {
    printf("Must specify an input bag!\n");
    exit(0);
  }
  ros::init(argc, argv, "lidar_slam");
  ros::NodeHandle n;
  signal(SIGINT, SignalHandler);
  // Load and pre-process the data.
  SLAMProblem2D slam_problem =
          ProcessBagFile(FLAGS_bag_path.c_str(), n);
  CHECK_GT(slam_problem.nodes.size(), 1)
    << "Not enough nodes were processed"
    << "you probably didn't specify the correct topics!\n";
  // Load all the residuals into the problem and run to get initial solution.
  Solver solver(FLAGS_translation_weight,
                FLAGS_rotation_weight,
                FLAGS_lc_translation_weight,
                FLAGS_lc_rotation_weight,
                FLAGS_stopping_accuracy,
                FLAGS_pose_output_file,
                n);
  LearnedLoopClosure(slam_problem, solver);
  std::cout << "Waiting for Loop Closure input" << std::endl;
  ros::Subscriber hitl_sub = n.subscribe(FLAGS_hitl_lc_topic,
                                         10,
                                         &Solver::HitlCallback,
                                         &solver);
  ros::Subscriber write_sub = n.subscribe("/write_output",
                                          10,
                                          &Solver::WriteCallback,
                                          &solver);
  ros::Subscriber vector_sub = n.subscribe("/vectorize_output",
                                           10,
                                           &Solver::Vectorize,
                                           &solver);
  ros::spin();
  return 0;
}
