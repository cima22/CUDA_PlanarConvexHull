#include <iostream>
#include <cmath>
#include "../points_generation/random_points.h"
#include <vector>
#include <chrono>
#include <algorithm>

// compute distance of a point from a line that passes by P and Q
double distance_from_line(const Point& p, const Point& q, const Point& v) {
    double a = q.y - p.y;
    double b = p.x - q.x;
    double c = (q.x * p.y) - (p.x * q.y);
    return std::abs((a * v.x + b * v.y + c) / std::sqrt(a * a + b * b));
}

enum Position {above,below,on};

// Check if a point is above the line PQ in a clockwise direction
Position isAboveClockwise(const Point& p, const Point& q, const Point& point) {
    double res = ((q.x - p.x) * (point.y - p.y)) - ((point.x - p.x) * (q.y - p.y));
    if(res > 0)
        return above;
    if(res < 0)
        return below;
    return on;
}

// Returns the index of the farthest point from segment (a, b).
Point getFarthest(const Point& a, const Point& b, const std::vector<Point>& v) {
    return *std::max_element(v.begin(), v.end(),
                             [&](const Point& p1, const Point& p2){
        return distance_from_line(a,b,p1) < distance_from_line(a,b,p2);
    });
}

// Recursive call of the quickhull algorithm.
void quickHull(const std::vector<Point>& v, const Point& a, const Point& b, std::vector<Point>& hull) {
    if (v.empty()) {
        return;
    }

    Point f = getFarthest(a, b, v);

    // Collect points to the left of segment (a, f)
    std::vector<Point> left;
    std::vector<Point> right;
    for (const auto p : v) {
        if (isAboveClockwise(a, f, p) == above) {
            left.push_back(p);
        }
        if (isAboveClockwise(f, b, p) == above) {
            right.push_back(p);
        }
    }
    hull.push_back(f);
    quickHull(left, a, f, hull);
    quickHull(right, f, b, hull);
}

// QuickHull algorithm
std::vector<Point> quickHull(const std::vector<Point>& v) {
    std::vector<Point> hull;
    auto by_x = [](const Point& p1, const Point& p2){
        return (p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y));
    };
    // Start with the leftmost and rightmost points.
    Point p = *std::min_element(v.begin(), v.end(), by_x);
    Point q = *std::max_element(v.begin(), v.end(), by_x);

    // Split the points on either side of segment (a, b)
    std::vector<Point> left, right;
    for (auto t : v) {
        switch (isAboveClockwise(p, q, t)) {
            case above:
                left.push_back(t);
                break;
            case below:
                right.push_back(t);
                break;
            case on:
                break;
        }
    }

    // Be careful to add points to the hull
    // in the correct order. Add our leftmost point.
    hull.push_back(p);

    // Add hull points from the left (top)
    quickHull(left, p, q, hull);

    // Add our rightmost point
    hull.push_back(q);

    // Add hull points from the right (bottom)
    quickHull(right, q, p, hull);

    return hull;
}

int main() {
    // generate the points in the plane
    auto points = generate_random_points();

    std::cout << N << " points randomly generated!" << std::endl;
	
     // Timer starts when first split function is called
    auto start = std::chrono::high_resolution_clock::now();
    auto chull = quickHull(points);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Found convex hull with " << chull.size() << " points.\nPrinting the first and last three points of the hull:" << std::endl;

    // Print first 3 elements
    for (int i = 0; i < 3; i++) {
	    printf("(%f, %f)\n", chull[i].x, chull[i].y);
	}
	
    std::cout << ".\n.\n." << std::endl;
	
    // Print last 3 elements
    for (int i = chull.size()-3; i < chull.size(); i++) {
	    printf("(%f, %f)\n",chull[i].x, chull[i].y);
    }

    // Compute time interval
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " ms" << std::endl;

    return 0;
}
