#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "../../../../Desktop/CUDA_PlanarConvexHull/src/points_generation/random_points.h"
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

// Function to execute the algorithm through CUDA
void quickHullCUDA(Point* points, int numPoints, int* hullPoints, int* numHullPoints) {
	Point left = points[0];
	Point right = points[0];
	// Find left-most and right-most points in the set
	for (int i = 1; i < numPoints; i++) {
    	if (points[i].x < left.x)
        	left = points[i];
    	if (points[i].x > right.x)
        	right = points[i];
	}

	// Set Block and Grid dimension
	int threadsPerBlock = 256;
	int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

	// Allocate points in the device memory
	Point* d_points;
	cudaMalloc((void**)&d_points, numPoints * sizeof(Point));
	cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);

	// Allocate space in the device for the hull
	int* d_hullPoints;
	int* d_numHullPoints;
	cudaMalloc((void**)&d_hullPoints, numPoints * sizeof(int));
	cudaMalloc((void**)&d_numHullPoints, sizeof(int));
	cudaMemset(d_numHullPoints, 0, sizeof(int));

	// Launch Kernels
	quickHullKernel<<<blocksPerGrid, threadsPerBlock>>>(d_points, numPoints, left, right, d_hullPoints, d_numHullPoints);
	quickHullKernel<<<blocksPerGrid, threadsPerBlock>>>(d_points, numPoints, right, left, d_hullPoints, d_numHullPoints);

	// Copy back from device the hull and its size
	cudaMemcpy(hullPoints, d_hullPoints, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(numHullPoints, d_numHullPoints, sizeof(int), cudaMemcpyDeviceToHost);

	// Free device space
	cudaFree(d_points);
	cudaFree(d_hullPoints);
	cudaFree(d_numHullPoints);
}

int main() {
	Point* points = (Point*)malloc(N * sizeof(Point));
	int* hullPoints = (int*)malloc(N * sizeof(int));
	int numHullPoints = 0;
	// Generate random points in the plane using the function from the generator
	vector<Point> r_points = generate_random_points();
	cout << N << " points randomly generated!" << endl;

	// From vector to array
	for (int i = 0; i < N; i++) {
    	points[i].x = r_points[i].x;
    	points[i].y = r_points[i].y;
	}
	// Timer starts
    	auto start = chrono::high_resolution_clock::now();
    
	// Find the convex hull
	quickHullCUDA(points, N, hullPoints, &numHullPoints);
	
	// Timer stops
	auto end = chrono::high_resolution_clock::now();

	cout << "Found convex hull with " << numHullPoints << " points.\nPrinting the first and last three points of the hull:" << endl;

	// Print first 3 elements
	for (int i = 0; i < 3; i++) {
    	printf("(%f, %f)\n", points[hullPoints[i]].x, points[hullPoints[i]].y);
	}
	
	cout << ".\n.\n." << endl;
	
	// Print last 3 elements
	for (int i = numHullPoints-3; i < numHullPoints; i++) {
    	printf("(%f, %f)\n",points[hullPoints[i]].x, points[hullPoints[i]].y);
	}

	// Compute time interval
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "Execution time: " << duration << " ms" << endl;
	
	// Free memory space
	free(points);
	free(hullPoints);

	return 0;
}
