CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(nautilus)

##########################
# Setting compiler flags #
##########################

SET(CMAKE_BUILD_TYPE Release)

# TODO: Remove no-deprecated-copy once Eigen 3.4 stable is released.
SET(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Werror -Wno-deprecated-copy")

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
    option(TESTING "Build tests" FALSE)
    MESSAGE(STATUS "Additional Flags for Release mode")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -O3 -DNDEBUG")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    option(TESTING "Build tests" TRUE)
    MESSAGE(STATUS "Additional Flags for Debug mode")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -g -O0")
ENDIF()

#####################
# Find Dependencies #
#####################

SET(Ceres_DIR {PROJECT_SOURCE_DIR}/third_party/ceres/ceres-solver-1.14.0/cmake)
FIND_PACKAGE(Ceres REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS system)
FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp rosbag std_msgs sensor_msgs geometry_msgs message_generation)

SET(third_party_includes ${CMAKE_CURRENT_SOURCE_DIR}/third_party/csm/src
                         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vector_maps/src
                         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/config-reader/include
                         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
                         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/learning-loop-closure
                         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ceres/ceres-solver-1.14.0/include)
######################
# Message Generation #
######################

add_message_files(DIRECTORY msg FILES CobotOdometryMsg.msg HitlSlamInputMsg.msg WriteMsg.msg)
generate_messages(DEPENDENCIES std_msgs sensor_msgs geometry_msgs)

########################
# Catkin Configuration #
########################

catkin_package(
  INCLUDE_DIRS ${third_party_includes} ${CERES_INCLUDE_DIRS}
  LIBRARIES amrl_shared
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs geometry_msgs)

INCLUDE_DIRECTORIES(include ${Boost_INCLUDE_DIRS}
                            ${catkin_INCLUDE_DIRS}
                            ${third_party_includes})
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/googletest/lib)

#####################
# Configure Targets #
#####################

# Make sure all our dependecies get built
add_subdirectory(third_party)

add_executable(${PROJECT_NAME} src/main.cc
                               src/pointcloud_helpers.cc
                               src/solver.cc
                               src/slam_type_builder.cc
                               src/kdtree.cc
                               src/normal_computation.cc)

add_library(amrl_shared src/gui_helpers.cc
                        src/timer.cc)

set(non_findable_libs gflags glog X11 m lua5.1 pthread csm vector_maps Boost::system stdc++)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES}
                                       amrl_shared
                                       ${CERES_LIBRARIES}
                                       ${OpenMP_CXX_LIBRARIES}
                                       ${non_findable_libs})

# We need the second Exported Targets because Laser_scan_matcher should be built first.
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} laser_scan_matcher_generate_messages_cpp local_uncertainty_matcher_generate_messages_cpp)

if (CATKIN_ENABLE_TESTING OR TESTING)
  MESSAGE(STATUS "Compiling tests")
  enable_testing()
  add_subdirectory(test)
endif()
