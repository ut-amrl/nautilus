PROJECT(lidar_slam)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

MESSAGE(STATUS "Compiling Lidar SLAM Implementation")
MESSAGE(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

SET(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Werror")

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
    option(TESTING "Build tests" OFF)
    MESSAGE(STATUS "Additional Flags for Release mode")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -O3 -DNDEBUG")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    option(TESTING "Build tests" ON)
    MESSAGE(STATUS "Additional Flags for Debug mode")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -g -O0")
ENDIF()

FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
FIND_PACKAGE(OpenMP REQUIRED)

INCLUDE($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
add_subdirectory(third_party)

INCLUDE_DIRECTORIES(third_party/csm/src)
INCLUDE_DIRECTORIES(third_party/vector_maps/src)
INCLUDE_DIRECTORIES(third_party/learning-loop-closure/laser_scan_matcher/srv_gen/cpp/include)
INCLUDE_DIRECTORIES(third_party/config-reader/include)

rosbuild_init()

SET(ROS_BUILD_STATIC_LIBS true)
SET(ROS_BUILD_SHARED_LIBS false)

MESSAGE(STATUS "ROS-Overrride Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

rosbuild_genmsg()

SET(libs roslib
         roscpp
         rosbag
         console_bridge
         cv_bridge
         gflags
         glog
         boost_system
         X11
         m
         OpenMP::OpenMP_CXX
         csm
         vector_maps
         lua5.1
         pthread)

ROSBUILD_ADD_EXECUTABLE(lidar_slam
                        src/main.cc
                        src/pointcloud_helpers.cc
                        src/solver.cc
                        src/slam_type_builder.cc
                        src/kdtree.cc
                        src/normal_computation.cc)

ADD_LIBRARY(shared_library
            src/gui_helpers.cc
            src/timer.cc)

TARGET_LINK_LIBRARIES(lidar_slam
                      shared_library
                      ${libs}
                      ${CERES_LIBRARIES})

#if (TESTING)
# enable_testing()
#add_subdirectory(test)
#endif()
