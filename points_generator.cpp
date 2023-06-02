//
// Created by Michele Tamborrino on 12/05/23.
//
#include <vector>
#include <random>
#include "random_points.h"

std::vector<Point> generate_random_points() {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-RANGE, RANGE);

    std::vector<Point> points(N);
    for (int i = 0; i < N; i++) {
        points[i].x = dist(rng);
        points[i].y = dist(rng);
    }

    return points;
}

