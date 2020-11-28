dofile(debug.getinfo(1).source:match("@?(.*/)") .. '/default_config.lua')
bag_path="data/ut-automata.bag"
lidar_topic="/scan"
odom_topic="/odom"
auto_lc=false
pose_number=1000
