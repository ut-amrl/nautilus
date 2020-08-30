require("config.default_config")
--bag_path="../bagfiles/2019-09-17-17-04-51.bag"
bag_path="../bagfiles/4_floor.bag"
lidar_topic="/scan"
odom_topic="/odom"
auto_lc=false
pose_number=1000
translation_change_for_lidar = 0.15
rotation_change_for_lidar = math.pi / 30

outlier_threshold=0.25
lidar_constraint_amount_min = 4
lidar_constraint_amount_max = 10

translation_scaling_1 = 1 / 50.0
translation_scaling_2 = 1 / 50.0
rotation_scaling_1 = 1 / 50.0
rotation_scaling_2 = 1 / 50.0
translation_standard_deviation = translation_scaling_1 * translation_change_for_lidar + rotation_scaling_1 * rotation_change_for_lidar
rotation_standard_deviation = translation_scaling_2 * translation_change_for_lidar + rotation_scaling_2 * rotation_change_for_lidar

-- rotation_weight = 10
-- translation_weight = 10
-- lidar_constraint_amount=1
-- hitl_line_width=0.1
