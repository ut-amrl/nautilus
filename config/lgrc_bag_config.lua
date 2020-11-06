dofile(debug.getinfo(1).source:match("@?(.*/)") .. '/default_config.lua')
bag_path="data/00010_2019-05-16-03-59-04_0.bag"
lidar_topic="/velodyne_2dscan_high_beams"
odom_topic="/odometry/filtered"
auto_lc=false
pose_number=350
accuracy_change_stop_threshold = 0.005
translation_weight=1.0
rotation_weight=1.0
lidar_constraint_amount_max=10
outlier_threshold=1
