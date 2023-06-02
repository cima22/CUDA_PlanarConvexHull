#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "random_points.h"
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// Function to compute distance between points
__device__ double distance(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Function to compute distance between a line PQ and a point
__device__ double distanceToLine(Point lineStart, Point lineEnd, Point point) {
    double lineLength = distance(lineStart, lineEnd);
    double area = 0.5f * fabsf(
        (lineEnd.x - lineStart.x) * (lineStart.y - point.y) - 
        (lineStart.x - point.x) * (lineEnd.y - lineStart.y)
    );
    return (2.0f * area) / lineLength;
}

// Check if a point is above the line PQ in a clockwise orientation
__device__ bool isAboveClockwise(Point p, Point q, Point point) {
	// Cross product gives which is the orientation
    double cross = ((q.x - p.x) * (point.y - p.y)) - ((point.x - p.x) * (q.y - p.y));
    return cross > 0;
}

// Kernel to compute the convex hull given a set of points in the plane
__global__ void quickHullKernel(Point* points, int numPoints, Point left, Point right, int* hullPoints, int* numHullPoints) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numPoints) return;

    double maxDistance = -1.0f;
    int maxIndex = -1;

    // Find farthest point
    for (int i = tid; i < numPoints; i += blockDim.x * gridDim.x) {
    	if (isAboveClockwise(left, right,  points[i])) {
        	double distance = distanceToLine(left, right, points[i]);
        	if (distance > maxDistance) {
            	maxDistance = distance;
            	maxIndex = i;
            }
        }
    }

    // Add the resulting point to the hull
    if (maxIndex >= 0) {
        int index = atomicAdd(numHullPoints, 1);
        hullPoints[index] = maxIndex;

        Point p = points[maxIndex];

        // Find left-most points
        if (numPoints > 0) {
            int innerIndex = atomicAdd(numHullPoints, 1);
            hullPoints[innerIndex] = tid;
        }
    }
}


int main() {
	Point* points;
	int* hullPoints;
	int* numHullPoints;
	// Generate random points in the plane using the function from the generator
	vector<Point> r_points = generate_random_points();
	cout << "Points randomly generated!" << endl;
	
	// Let's use Unified Memory
	cudaMallocManaged((void**)&points, N*sizeof(Point)); // input
	cudaMallocManaged((void**)&hullPoints, N*sizeof(int)); // output
	cudaMallocManaged(&numHullPoints, sizeof(int)); // output size
	*numHullPoints = 0; // first output size
	
	// From vector to array
	for (int i = 0; i < N; i++) {
    	points[i].x = r_points[i].x;
    	points[i].y = r_points[i].y;
	}
	
	Point left = points[0];
	Point right = points[0];
	// Find left-most and right-most points in the set
	for (int i = 1; i < N; i++) {
    	if (points[i].x < left.x)
        	left = points[i];
    	if (points[i].x > right.x)
        	right = points[i];
	}

	// Set Block and Grid dimension
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	// Timer starts
    auto start = chrono::high_resolution_clock::now();
    
	// Launch Kernels
	quickHullKernel<<<blocksPerGrid, threadsPerBlock>>>(points, N, left, right, hullPoints, numHullPoints);
	quickHullKernel<<<blocksPerGrid, threadsPerBlock>>>(points, N, right, left, hullPoints, numHullPoints);
	// synchronize
	cudaDeviceSynchronize();
	
	// Timer stops
	auto end = chrono::high_resolution_clock::now();

	// Print first 3 elements
	for (int i = 0; i < 3; i++) {
    	printf("Hull Point %d: (%f, %f)\n", i, points[hullPoints[i]].x, points[hullPoints[i]].y);
	}
	
	cout << ".\n.\n." << endl;
	
	// Print last 3 elements
	for (int i = *numHullPoints-3; i < *numHullPoints; i++) {
    	printf("Hull Point %d: (%f, %f)\n", i, points[hullPoints[i]].x, points[hullPoints[i]].y);
	}

	// Compute time interval
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "Execution time: " << duration << " ms" << endl;
	
	// Free memory space
	cudaFree(points);
	cudaFree(hullPoints);

	return 0;
}