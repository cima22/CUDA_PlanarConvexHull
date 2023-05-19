#include <iostream>
#include <cmath>
#include "random_points.h"
#include <vector>
#include <chrono>
#include <algorithm>

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

// Check if a point is on the left of another
bool isLeftOf(const Point& a, const Point& b) {
    return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

// Returns the index of the farthest point from segment (a, b).
size_t getFarthest(const Point& a, const Point& b, const vector<Point>& v) {
    size_t idxMax = 0;
    double distMax = distance_from_line(a, b, v[idxMax].x, v[idxMax].y);

    for (size_t i = 1; i < v.size(); ++i) {
        double distCurr = distance_from_line(a, b, v[i].x, v[i].x);
        if (distCurr > distMax) {
            idxMax = i;
            distMax = distCurr;
        }
    }

    return idxMax;
}

// Recursive call of the quickhull algorithm.
void quickHull(const vector<Point>& v, const Point& a, const Point& b,vector<Point>& hull) {

    if (v.empty()) {
        return;
    }

    Point f = v[getFarthest(a, b, v)];

    // Collect points to the left of segment (a, f)
    vector<Point> left;
    for (auto p : v) {
        if (isAboveClockwise(a, f, b)) {
            left.push_back(p);
        }
    }
    quickHull(left, a, f, hull);

    // Add f to the hull
    hull.push_back(f);

    // Collect points to the left of segment (f, b)
    vector<Point> right;
    for (auto p : v) {
        if (isAboveClockwise(f, b, p)) {
            right.push_back(p);
        }
    }
    quickHull(right, f, b, hull);
}

// QuickHull algorithm
vector<Point> quickHull(const vector<Point>& v) {
    vector<Point> hull;

    // Start with the leftmost and rightmost points.
    Point p = *min_element(v.begin(), v.end(), isLeftOf);
    Point q = *max_element(v.begin(), v.end(), isLeftOf);

    // Split the points on either side of segment (a, b)
    vector<Point> left, right;
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

    cout << "Points generated!" << endl;

    // allocate vector to contain the points of the convex hull
    std::vector<Point> chull;

    cout << "Beginning (serial) Quickhull algorithm..." << endl;
    // Timer starts when first split function is called
    auto start = chrono::high_resolution_clock::now();
    chull = quickHull(points);

    // Print the result vector
    for (const auto& point : chull) {
        cout << point.x << " "<< point.y << endl;
    }

    cout << chull.size() << endl;

    auto end = chrono::high_resolution_clock::now();
    // Compute time interval
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "Execution time: " << duration << " ms" << endl;
    return 0;
}
