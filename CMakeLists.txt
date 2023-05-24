cmake_minimum_required(VERSION 3.25)
project(gpu)

set(CMAKE_CXX_STANDARD 20)

add_executable(gpu serial_quickhull.cpp random_points.h points_generator.cpp)
