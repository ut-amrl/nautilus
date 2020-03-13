# Should this be done with a roslaunch script? Yes,
# Will it be? No.

cd third_party/learning-loop-closure/point_cloud_embedder
rosrun point_cloud_embedder point_cloud_embedder.py ../../../data/model_embedder.pth
