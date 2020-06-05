# LIDAR SLAM

This project aims to be a professional point-to-point slam implementation.

### Downloading:

Because this project uses submodules for external libraries please clone using the following command.

```
git clone --recurse-submodules <git project URL>
```

### Dependencies:

You will need to install GTest for testing. This can be done by first installing gtest through apt.

On Ubuntu:

```sudo apt-get install libgtest-dev```

You will also need Lua 5.1 and clang for the config-reader library. You can get these by running the InstallPackages script in ```third_party/config-reader```.

### Compiling
And then navigate to the installed directory and make the project using the following code:

```
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo make install
```
(Here is the tutorial this is from: https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/)

Then run:

```
cd third_party/learning-loop-closure/laser_scan_matcher
make
cd ../../..
make
```

Do not run cmake directly.

### How to run:

If you are using auto loop closure detection and solving (on by default) you will have to use the following command to start the embedding generation node.
```
<Starting in the root of the project>
cd third_party/learning-loop-closure/laser_scan_matcher
source build/devel/setup.sh
rosrun laser_scan_matcher laser_scan_matcher.py <model filename>
```

Then in another terminal window continue the following instructions.


On linux:
```./bin/lidar_slam --config_file <config_file you want to use>```

### Examples:

Here are three bagfiles that you can run the data on and the command to run them:

[Indoor Bag taken on the "Jackal"](https://drive.google.com/open?id=1thDp4MJF6l2yZ9Z_JFAmdhMQZrld0oQ5)

```
./bin/lidar_slam --config_file lgrc_bag_config.lua
```

[Indoor Bag taken on the "COBOT"](https://drive.google.com/open?id=1i7RlzAbIoVkKpZGa7TcJaO3kzSf7KI3D)

```
./bin/lidar_slam --config_file cmu_cobot_config.lua
```

[Indoor Bag taken on the "COBOT" in the GDC](https://drive.google.com/a/utexas.edu/file/d/1KXN9eDzBZAnd34Nr30useKH3JP7-xVxL/view?usp=drivesdk)
```
./bin/lidar_slam --config_fle gdc_2020_config.lua
```

