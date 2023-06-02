//
// Created by Michele Tamborrino on 12/05/23.
//
#include <vector>
#include <random>
#include "random_points.h"

double Point::distance_from(const Point& b){
	double dx = x - b.x;
	double dy = y - b.y;
	return std::sqrt(dx * dx + dy * dy);
}

std::vector<Point> generate_random_points() {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-RANGE, RANGE);
/*
    std::vector<Point> points(N);
    for (int i = 0; i < N; i++) {
        points[i].x = dist(rng);
        points[i].y = dist(rng);
    }
    */
    return {Point(1, 1),Point(2, 2),Point(4, 6),Point(5, 3),Point(6, 3),Point(7, 2),Point(9, 1)};

    //return points;
}

