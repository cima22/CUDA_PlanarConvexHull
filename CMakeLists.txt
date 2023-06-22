cmake_minimum_required(VERSION 3.25)
project(gpu)

set(CMAKE_CXX_STANDARD 20)

add_executable(gpu src/quickhull/serial_quickhull.cpp src/points_generation/random_points.h src/points_generation/points_generator.cpp src/quickhull/thurst_quickhull.cu)
