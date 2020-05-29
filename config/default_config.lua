--[[
    This is the default configuration for all config files
    using the lidar_slam project.

    Add the following to the top of every config file:
    require("config.default_config.lua")

    Please do not modify any of these values here as they are proven
    to work. If you want a different value for any of them, change it
    in your own config file by giving it a new value.
]]--

--[[ I/O Variables and General Problem Variables ]]--

-- The path from the root of the project to a ROS bag file.
bag_path=""

-- The number of lidar scans (and poses) to load from the bag file.
pose_number=30

-- The ROS topic that contains nav_msgs/Odometry messages.
odom_topic=""

-- The ROS topic that contains sensor_msgs/LaserScan messages.
lidar_topic=""

-- True if the messages are differential (/Cobot/Odometry messages)
-- AMRL specific.
differential_odom=false

-- The filename to output the finished poses to.
pose_output_file="poses.txt"


--[[ Performance Tuning Variables ]]--

-- The translation multiplier used in the odometry residuals.
translation_weight=1

-- The rotation multiplier used in the rotation residuals.
rotation_weight=1

-- If the scans change by an amount less than this during an entire
-- minimization iteration then the problem is deemed solved.
stopping_accuracy=0.05

-- Any point past this range in the lidar scan is truncated out.
max_lidar_range=30

-- Loop closure translation multiplier, used during loop closure for
-- odometry residuals.
lc_translation_weight=1

-- Loop closure rotation multiplier, used during loop closure for
-- odometry residuals.
lc_rotation_weight=1

-- The amount of rotation needed to signal a lidar scan capture.
rotation_change_for_lidar=math.pi / 18

-- The amount of translation needed to signal a lidar scan capture.
translation_change_for_lidar=0.25

-- The amount of previous lidar scans that each lidar scan will be compared against.
lidar_constraint_amount=10

-- Points further than this distance from each other cannot be counted
-- as the same point during ICL / ICP.
outlier_threshold=0.25

-- Translational and Rotational standard deviation, proportional to the translational and rotational change.
-- See these slides for more information:
-- https://docs.google.com/presentation/d/1BNHQwS6eHec8QiOcSBBNpA3mrDqHAYvrXJGju4lLUjU/edit#slide=id.g80de3824c7_0_308
-- Scaling factors used in calculation.
translation_scaling_1 = 1 / 10
translation_scaling_2 = 1 / 10
rotation_scaling_1 = 1 / 20
rotation_scaling_2 = 1 / 20
translation_standard_deviation = translation_scaling_1 * translation_change_for_lidar + rotation_scaling_1 * rotation_change_for_lidar
rotation_standard_deviation = translation_scaling_2 * translation_change_for_lidar + rotation_scaling_2 * rotation_change_for_lidar

--[[ HITL LC Variables ]]--

-- HITL LC topic, tools that send HitlSlamInputMsg publish on this topic.
hitl_lc_topic="/hitl_slam_input"

-- Above this threshold and the CSM transformation is deemed successful.
csm_score_threshold=-5.0

-- Points further than this will not count as falling on the HITL LC line.
hitl_line_width=0.05

-- The amount of points needed to count a scan as on the HITL LC line.
hitl_pose_point_threshold=10

--[[ Automatic LC Variables ]]--

-- Automatically loop close or not
auto_lc=false

-- All scans with local uncertainty less than this threshold are
-- one step closer to being used for automatic lc.
local_uncertainty_condition_threshold=9.5

-- All scans with local uncertainty scale less than this threshold
-- are one step closer to being used for automatic lc.
local_uncertainty_scale_threshold=0.3

-- The amount of previous scans to use for calculating local uncertainty.
local_uncertainty_prev_scans=2

-- threshold used in automatic loop closure.
lc_match_threshold=0.5

-- base max range to consider a loop closure
lc_base_max_range = 3.0

-- how much max range to consider a loop closure increases as nodes get more distant
lc_max_range_scaling = 0.05

-- used to dump images from auto-lc
lc_debug_output_dir="auto_lc_debug"

--[[ Normal Computation Variables ]]--

-- The neighborhood size to consider when RANSACing for normals.
nc_neighborhood_size=0.15

-- How much the neighborhood increases with each iteration.
nc_neighborhood_step_size=0.1

-- You got me, this one's just a constant.
nc_mean_distance=0.1

-- The number of buckets to use in the Hough transform.
nc_bin_number=32