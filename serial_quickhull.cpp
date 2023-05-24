#include <iostream>
#include <cmath>
#include "random_points.h"
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

// Check if a point is above the line PQ in a clockwise direction
bool isAboveClockwise(const Point& p, const Point& q, const Point& point) {
    return (((q.x - p.x) * (point.y - p.y)) - ((point.x - p.x) * (q.y - p.y))) > 0;
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
    for (const auto p : v) {
        if (isAboveClockwise(a, f, p)) {
            left.push_back(p);
        }
    }
    quickHull(left, a, f, hull);

    // Add f to the hull
    hull.push_back(f);

    // Collect points to the left of segment (f, b)
    std::vector<Point> right;
    for (const auto p : v) {
        if (isAboveClockwise(f, b, p)) {
            right.push_back(p);
        }
    }
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
        isAboveClockwise(p, q, t) ? left.push_back(t) : right.push_back(t);
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
    std::vector<Point> points(N);
    points = generate_random_points();

    std::cout << "Points generated!" << std::endl;
	
    for(const auto& point : points)
	    std::cout << point.x << " " << point.y << "\n";

    // allocate vector to contain the points of the convex hull
    std::vector<Point> chull;

    std::cout << "\n\nBeginning (serial) Quickhull algorithm..." << std::endl;
    // Timer starts when first split function is called
    auto start = std::chrono::high_resolution_clock::now();
    chull = quickHull(points);

    // Print the result vector
    for (const auto& point : chull) {
	    std::cout << point.x << " "<< point.y << std::endl;
    }

    std::cout << chull.size() << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    // Compute time interval
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << duration << " ms" << std::endl;
    return 0;
}
