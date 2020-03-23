# Should this be done with a roslaunch script? Yes,
# Will it be? No.

cd third_party/learning-loop-closure/point_cloud_embedder
export ROS_PACKAGE_PATH=$(pwd):$ROS_PACKAGE_PATH
rosrun point_cloud_embedder point_cloud_embedder.py $1
