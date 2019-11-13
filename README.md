# LIDAR SLAM

This project aims to be a professional point-to-point slam implementation.

### Compiling:

Just run:
```make```

Do not run cmake directly.

### How to run:

On linux:
```./bin/lidar_slam --bag_path <bags>```

### Arguments:

- ```--odom_topic <name>``` this will look for odometry messages on the ROS topic named ```<name>```.
- ```--lidar_topic <name>``` this will look for lidar messages on the ROS topic named ```<name>```.
- ```--pose_num <num>``` this will set the number of poses to process to ```<num>```.
- ```--stopping_accuracy <num>``` stops the joint optimization once the net change between points is less than ```<num>```.
- ```--rotation_weight <num>``` the weight multiplier for changing the odometry predicted rotation between poses.
- ```--translation_weight <num>``` the weight multiplier for changing the odometry predicted translation between poses.

### Examples:

Here are three bagfiles that you can run the data on and the command to run them:

[Indoor Bag taken on the "Jackal"](https://drive.google.com/open?id=1thDp4MJF6l2yZ9Z_JFAmdhMQZrld0oQ5)

```
./bin/lidar_slam --bag_path 00010_2019-05-16-03-59-04_0.bag --pose_num 1000
```

[Outdoor Bag taken on the "Jackal"](https://drive.google.com/open?id=1iLCKV4nnVvCzDQS2EHKotdTxiWPOCW-I)

```
./bin/lidar_slam --bag_path 00016_2019-05-17-18-23-06_0.bag --pose_num 1000
```

[Indoor Bag taken on the "COBOT"](https://drive.google.com/open?id=1i7RlzAbIoVkKpZGa7TcJaO3kzSf7KI3D)

```
./bin/lidar_slam --bag_path 2014-10-07-12-58-30.bag --odom_topic odom --lidar_topic laser --pose_num 1000
```

[Indoor Bag taken on the "COBOT" in the GDC](https://drive.google.com/a/utexas.edu/file/d/1KXN9eDzBZAnd34Nr30useKH3JP7-xVxL/view?usp=drivesdk)
```
./bin/lidar_slam --bag_path data/2019-11-08-11-13-09_GDC3.bag --lidar_topic /Cobot/Laser --odom_topic /Cobot/Odometry --diff_odom --pose_num 1000 --translation_weight 20
```

