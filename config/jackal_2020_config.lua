dofile(debug.getinfo(1).source:match("@?(.*/)") .. '/default_config.lua')
bag_path="data/ut-ahg-eer_2020-09-12-17-18-38.bag"
lidar_topic="/scan"
odom_topic="/jackal_velocity_controller/odom"
auto_lc=false
pose_number=1200
translation_weight=1
rotation_weight=1
hitl_line_width=.10
