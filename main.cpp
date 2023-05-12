#include <iostream>
#include <cmath>
#include "random_points.h"
#include <vector>
#include <chrono>


using namespace std;

// compute distance between points
double distance(const Point& p1, const Point& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx * dx + dy * dy);
    return distance;
}

// compute distance of a point from a line that passes by P and Q
double distance_from_line(const Point& p, const Point& q, double x, double y) {
    double a = q.y - p.y;
    double b = p.x - q.x;
    double c = (q.x * p.y) - (p.x * q.y);

    double distance = std::abs((a * x + b * y + c) / std::sqrt(a * a + b * b));
    return distance;
}

// Check if a point is above the line PQ in a clockwise direction
bool isAboveClockwise(const Point& p, const Point& q, const Point& point) {
    double cross = ((q.x - p.x) * (point.y - p.y)) - ((point.x - p.x) * (q.y - p.y));
    return cross > 0;
}


void split(const Point& p, const Point& q, const vector<Point>& points, vector<Point>& result, int& size) {

    if (size == N)
        return;

    // assume the first element is the farthest from the line
    Point max_point = points[0];
    double max_dist = distance_from_line(p, q, points[0].x, points[0].y);

    // Iterate over all points anche check whether they are above the line (clockwise)
    for (const auto& point : points) {
        if (isAboveClockwise(p, q, point)) {
            // compute the distance between the current point and the line pq
            double dist = distance_from_line(p, q, point.x, point.y);
            // if the distance is greater than the maximum, update it
            if (dist > max_dist) {
                max_point = point;
                max_dist = dist;
            }
        }
    }

    result.push_back(max_point);
    size++;
    split(p, max_point, points, result, size);
    split(max_point, q, points, result, size);
}

int main() {
    // generate the points in the plane
    std::vector<Point> points(N);
    points = generate_random_points();

    cout << "Points generated!" << endl;

    // allocate vector to contain the points of the convex hull
    std::vector<Point> chull(N);
    int hullsize = 0;

    // example starting points that are definetely on the hull (they can be changed)
    Point p = {-24900.9, -4595.72};
    Point q = {22318.7, 6684.25};

    cout << "Beginning Quickhull algorithm..." << endl;
    // Timer starts when first split function is called
    auto start = chrono::high_resolution_clock::now();
    split(p, q, points, chull, hullsize);

    /* Print the result vector
    for (const auto& point : chull) {
        cout << point.x << " "<< point.y << endl;
    }

    */

    auto end = std::chrono::high_resolution_clock::now();
    // Compute time interval
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "Execution time: " << duration << " ms" << endl;
    return 0;
}
