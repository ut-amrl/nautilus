#include <csignal>

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

using std::string;
using std::vector;

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

slam_types::SLAMProblem2D ProcessBagFile(const char* bag_path,
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
  } catch (rosbag::BagException& exception) {
    printf("Unable to read %s, reason %s:", bag_path, exception.what());
    return slam_types::SLAMProblem2D();
  }
  // Get the topics we want
  vector<string> topics;
  topics.emplace_back(FLAGS_odom_topic.c_str());
  topics.emplace_back(FLAGS_lidar_topic.c_str());
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  SLAMTypeBuilder slam_builder(FLAGS_pose_num);
  // Iterate through the bag
  for (rosbag::View::iterator it = view.begin();
       ros::ok() && it != view.end();
       ++it) {
    const rosbag::MessageInstance &message = *it;
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
  }
  bag.close();
  printf("Done.\n");
  fflush(stdout);
  return slam_builder.GetSlamProblem();
}

void SignalHandler(int signum) {
  printf("Exiting with %d\n", signum);
  exit(0);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(*argv);
  google::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_bag_path == "") {
    printf("Must specify an input bag!");
    exit(0);
  }
  ros::init(argc, argv, "lidar_slam");
  ros::NodeHandle n;
  signal(SIGINT, SignalHandler);
  // Load and pre-process the data.
  slam_types::SLAMProblem2D slam_problem =
          ProcessBagFile(FLAGS_bag_path.c_str(), n);
  CHECK_GT(slam_problem.nodes.size(), 1)
    << "Not enough nodes were processed"
    << "you probably didn't specify the correct topics!\n";
  // Load all the residuals into the problem and run!
  Solver solver(FLAGS_translation_weight,
                FLAGS_rotation_weight,
                FLAGS_stopping_accuracy);
  solver.SolveSLAM(slam_problem, n);
  return 0;
}
