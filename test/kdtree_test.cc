//
// Created by jack on 7/3/20.
//

#include <memory>

#include "Eigen/Dense"
#include "gtest/gtest.h"

#include "../src/kdtree.h"

using Eigen::Vector2f;
using std::vector;

TEST(KDTreeFloatTest, build_test) {
    config_reader::ConfigReader reader({"config/default_config.lua"});
    vector<Vector2f> point_vector = {Vector2f(0, 0), Vector2f(1, 0), Vector2f(-1, 0), Vector2f(0, 1), Vector2f(0, -1)};
    auto* tree = new KDTree<float, 2>(point_vector);
    // Just check to see if all these points made it into the tree.
    vector<std::shared_ptr<KDNodeValue<float, 2>>> neighbors;
    tree->FindNeighborPoints(Vector2f(0, 0), 3, &neighbors);
    delete tree;
    EXPECT_EQ(neighbors.size(), 5);
}

TEST(KDTreeFloatTest, min_tree_test) {
  config_reader::ConfigReader reader({"config/default_config.lua"});
  vector<Vector2f> point_vector = {Vector2f(0, 0), Vector2f(1, 0), Vector2f(-1, 0), Vector2f(0, 1), Vector2f(0, -1)};
  auto* tree = new KDTree<float, 2>(point_vector);
  // Just check to see if all these points made it into the tree.
  vector<std::shared_ptr<KDNodeValue<float, 2>>> neighbors;
  tree->FindNeighborPoints(Vector2f(0, 0), 3, &neighbors);
  auto min_value_x = tree->GetMinimumValue(0);
  auto min_value_y = tree->GetMinimumValue(1);
  delete tree;
  EXPECT_EQ(neighbors.size(), 5);
  EXPECT_FLOAT_EQ(min_value_x->point.x(), -1);
  EXPECT_FLOAT_EQ(min_value_y->point.y(), -1);
  point_vector = {Vector2f(0.2, 0.6), Vector2f(0.7, 0.87), Vector2f(0.12, 0.9)};
  tree = new KDTree<float, 2>(point_vector);
  min_value_x = tree->GetMinimumValue(0);
  min_value_y = tree->GetMinimumValue(1);
  delete tree;
  EXPECT_FLOAT_EQ(min_value_x->point.x(), 0.12);
  EXPECT_FLOAT_EQ(min_value_y->point.y(), 0.6);
  // Now try on an empty tree. Should return nullptr.
  tree = new KDTree<float, 2>(vector<Vector2f>());
  min_value_x = tree->GetMinimumValue(0);
  min_value_y = tree->GetMinimumValue(1);
  delete tree;
  EXPECT_EQ(min_value_x, nullptr);
  EXPECT_EQ(min_value_y, nullptr);
  // Now on just the root. Should just return the root.
  tree = new KDTree<float, 2>({Vector2f(-1, -1)});
  min_value_x = tree->GetMinimumValue(0);
  min_value_y = tree->GetMinimumValue(1);
  delete tree;
  EXPECT_FLOAT_EQ(min_value_x->point.x(), -1);
  EXPECT_FLOAT_EQ(min_value_y->point.y(), -1);
}

TEST(KDTreeFloatTest, max_tree_test) {
  config_reader::ConfigReader reader({"config/default_config.lua"});
  vector<Vector2f> point_vector = {Vector2f(0, 0), Vector2f(1, 0), Vector2f(-1, 0), Vector2f(0, 1), Vector2f(0, -1)};
  auto* tree = new KDTree<float, 2>(point_vector);
  // Just check to see if all these points made it into the tree.
  vector<std::shared_ptr<KDNodeValue<float, 2>>> neighbors;
  tree->FindNeighborPoints(Vector2f(0, 0), 3, &neighbors);
  auto min_value_x = tree->GetMaximumValue(0);
  auto min_value_y = tree->GetMaximumValue(1);
  delete tree;
  EXPECT_EQ(neighbors.size(), 5);
  EXPECT_FLOAT_EQ(min_value_x->point.x(), 1);
  EXPECT_FLOAT_EQ(min_value_y->point.y(), 1);
  point_vector = {Vector2f(0.2, 0.6), Vector2f(0.7, 0.87), Vector2f(0.12, 0.9)};
  tree = new KDTree<float, 2>(point_vector);
  min_value_x = tree->GetMaximumValue(0);
  min_value_y = tree->GetMaximumValue(1);
  delete tree;
  EXPECT_FLOAT_EQ(min_value_x->point.x(), 0.7);
  EXPECT_FLOAT_EQ(min_value_y->point.y(), 0.9);
  // Now try on an empty tree. Should return nullptr.
  tree = new KDTree<float, 2>(vector<Vector2f>());
  min_value_x = tree->GetMaximumValue(0);
  min_value_y = tree->GetMaximumValue(1);
  delete tree;
  EXPECT_EQ(min_value_x, nullptr);
  EXPECT_EQ(min_value_y, nullptr);
  // Now on just the root. Should just return the root.
  tree = new KDTree<float, 2>({Vector2f(-1, -1)});
  min_value_x = tree->GetMaximumValue(0);
  min_value_y = tree->GetMaximumValue(1);
  delete tree;
  EXPECT_FLOAT_EQ(min_value_x->point.x(), -1);
  EXPECT_FLOAT_EQ(min_value_y->point.y(), -1);
}

TEST(KDTreeFloatTest, remove_tree_test) {
  config_reader::ConfigReader reader({"config/default_config.lua"});
  vector<Vector2f> point_vector = {Vector2f(0, 0), Vector2f(1, 0), Vector2f(-1, 0), Vector2f(0, 1), Vector2f(0, -1)};
  auto tree = std::make_shared<KDTree<float, 2>>(point_vector);
  // Just check to see if all these points made it into the tree.
  vector<std::shared_ptr<KDNodeValue<float, 2>>> neighbors;
  tree->RemoveNearestValue(Vector2f(0, 0), 0.001);
  tree->FindNeighborPoints(Vector2f(0, 0), 0.001, &neighbors);
  EXPECT_EQ(neighbors.size(), 0);
  // A little more intricate of a test.
  point_vector = {Vector2f(0.2, 0.7), Vector2f(1.1, 0.9), Vector2f(-1.7, 4.4)};
  tree = std::make_shared<KDTree<float, 2>>(point_vector);
  neighbors.clear();
  vector<std::shared_ptr<KDNodeValue<float, 2>>> whole_tree;
  tree->RemoveNearestValue(Vector2f(0.2, 0.7), 0.001);
  tree->FindNeighborPoints(Vector2f(0, 0), 0.001, &neighbors);
  EXPECT_EQ(neighbors.size(), 0);
  tree->RemoveNearestValue(Vector2f(-1.7, 4.4), 0.001);
  tree->FindNeighborPoints(Vector2f(-1.7, 4.4), 0.001, &neighbors);
  EXPECT_EQ(neighbors.size(), 0);
  tree->RemoveNearestValue(Vector2f(1.1, 0.9), 0.001);
  tree->FindNeighborPoints(Vector2f(1.1, 0.9), 0.001, &neighbors);
  EXPECT_EQ(neighbors.size(), 0);
  tree->FindNeighborPoints(Vector2f(0, 0), 100, &whole_tree);
  EXPECT_EQ(neighbors.size(), 0);
  EXPECT_EQ(whole_tree.size(), 0);
  // Now lets test with a single node.
  point_vector = {Vector2f(200, 10)};
  tree = std::make_shared<KDTree<float, 2>>(point_vector);
  tree->RemoveNearestValue(Vector2f(200, 10), 0.001);
  tree->FindNeighborPoints(point_vector[0], 1000, &neighbors);
  EXPECT_EQ(neighbors.size(), 0);
  // This finally should just be ok.
  tree = std::make_shared<KDTree<float, 2>>(vector<Vector2f>());
  tree->RemoveNearestValue(Vector2f(0, 0), 0.001);
}