dofile(debug.getinfo(1).source:match("@?(.*/)") .. '/default_config.lua')
bag_path="data/00010_2019-05-16-03-59-04_0.bag"
lidar_topic="/velodyne_2dscan_high_beams"
odom_topic="/odometry/filtered"
auto_lc=false
pose_number=450
