#!/bin/bash

clang-format-9 -i -style=Google src/input/*
clang-format-9 -i -style=Google src/optimization/*
clang-format-9 -i -style=Google src/util/*
clang-format-9 -i -style=Google src/visualization/*
clang-format-9 -i -style=Google src/main.cc
