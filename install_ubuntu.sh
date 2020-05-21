#!/bin/bash
# Check for super user priviledges.
if [ $(whoami) != root ]
then
  echo "You must be root to install the dependencies."
  echo "Please rerun the script using sudo."
  exit
fi
SCRIPT_PATH=$(realpath -s $0)
SCRIPT_DIR=$(dirname $SCRIPT_PATH)
THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"
# These are things that can be installed using apt.
echo -e "\e[32mInstalling General APT Dependencies\e[39m"
apt install cmake libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev libgtest-dev curl libomp-dev
# Install GTest
echo -e "\e[32mInstalling GTest\e[39m"
cd /usr/src/gtest
mkdir build
cd build
cmake ..
make
# Install Eigen
echo -e "\e[32mDownloading Eigen\e[39m"
cd $THIRD_PARTY_DIR
EIGEN_TAR=eigen.tar.gz
curl -o $EIGEN_TAR https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
tar zxf $EIGEN_TAR
mv eigen-3.3.7 eigen
rm $EIGEN_TAR
# Install dependencies for ConfigReader
echo -e "\e[32mInstalling Config-Reader Dependencies\e[39m"
if [ -e "$THIRD_PARTY_DIR/config-reader/InstallPackages" ]
then
  bash $THIRD_PARTY_DIR/config-reader/InstallPackages
else 
  echo -e "\e[32mInstall script missing for ConfigReader, did you clone with submodules?\e[39m"
fi
# Install Ceres
echo -e "\e[32mInstalling Ceres\e[39m"
cd $THIRD_PARTY_DIR
CERES_TAR=ceres.tar.gz
curl -o $CERES_TAR http://ceres-solver.org/ceres-solver-1.14.0.tar.gz
tar zxf ceres.tar.gz
mkdir -p ceres
cd ceres
cmake ../ceres-solver-1.14.0
make -j $(nproc)
make install
cd ..
mv ceres-solver-1.14.0 ceres/ceres-solver-1.14.0
rm $CERES_TAR

