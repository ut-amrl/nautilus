dofile(debug.getinfo(1).source:match("@?(.*/)") .. '/default_config.lua')
bag_path="data/synthetic-small.bag"
lidar_topic="/velodyne_2dscan_high_beams"
odom_topic="/odometry/filtered"
auto_lc=false
pose_number=30
accuracy_change_stop_threshold = 0.0001
translation_weight=1
rotation_weight=1
