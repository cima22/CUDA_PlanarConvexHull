//
// Created by Michele Tamborrino on 14/05/23.
//
#include <iostream>
#include <cmath>
#include "random_points.h"
#include <vector>
#include <chrono>

using namespace std;

int main() {
    // generate the points in the plane
    vector<Point> points(N);
    points = generate_random_points();

    cout << "Points generated!" << endl;

    // allocate vector to contain the points of the convex hull
    vector<Point> chull(N);
    int hullsize = 0;
    // Timer starts when first split function is called
    auto start = chrono::high_resolution_clock::now();

    auto end = chrono::high_resolution_clock::now();
    // Compute time interval
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "Execution time: " << duration << " ms" << endl;

    return 0;
}