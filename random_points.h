//
// Created by Michele Tamborrino on 12/05/23.
//

#ifndef GPU_RANDOM_POINTS_H
#define GPU_RANDOM_POINTS_H

#include <vector>

struct Point {
    double x;
    double y;

    double distance_from(const Point& b);
};

std::vector<Point> generate_random_points();

const int N = 5;
const int RANGE = 10;

static unsigned int seed = 2023;

#endif //GPU_RANDOM_POINTS_H
