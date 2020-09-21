dofile(debug.getinfo(1).source:match("@?(.*/)") .. '/default_config.lua')
bag_path="data/2020-03-09-19-02-17-GDC3-long.bag"
lidar_topic="/Cobot/Laser"
odom_topic="/Cobot/Odometry"
auto_lc=false
pose_number=1000
differential_odom=true
rotation_weight=1
translation_weight=2
max_lidar_range=8.5
