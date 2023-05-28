#include <iostream>
#include <cmath>
#include "random_points.h"
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>


// Check if a point is above the line PQ in a clockwise orientation
__device__ bool isAboveClockwise(Point p, Point q, Point point) {
	// Cross product gives which is the orientation
    double cross = ((q.x - p.x) * (point.y - p.y)) - ((point.x - p.x) * (q.y - p.y));
    return cross > 0;
}

// Functor to compute distance between a line PQ and a point
struct DistanceToLineFunctor {
    Point lineStart;
    Point lineEnd;

    __device__
    double operator()(Point point) const {
        double lineLength = distance(lineStart, lineEnd);
        double area = 0.5f * fabsf(
            (lineEnd.x - lineStart.x) * (lineStart.y - point.y) - 
            (lineStart.x - point.x) * (lineEnd.y - lineStart.y)
        );
        return (2.0f * area) / lineLength;
    }

    // Funzione ausiliaria per calcolare la distanza tra due punti
    __device__
    double distance(Point p1, Point p2) const {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        return sqrt(dx * dx + dy * dy);
    }
};

__global__ void quickhull(const thrust::device_vector<Point>& points, int left, int right, std::vector<int>& convex_Hull) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	
    if (tid >= N) return;
    
    double maxDistance = -1.0f;
    int maxIndex = -1;
    
    // vector of points above the line
    
}

int main() {
	
	// Create vector of point in the host
	thrust::host_vector<Point> host_Points = generate_random_points();
	
	// Print first 20 elements
	for (int i = 0; i < 20; i++) {
    	printf("Point %d: (%f, %f)\n", i, host_Points[i].x, host_Points[i].y);
	}
	
	thrust::host_vector<double> h_x;
    thrust::host_vector<double> h_y;
    
    // Decouple vector points in coordinates vectors
	
	for (const auto& point : host_Points) {
        h_x.push_back(point.x);
        h_y.push_back(point.y);
    }
    
    // Copy from host to device
    thrust::device_vector<double> d_x = h_x;
    thrust::device_vector<double> d_y = h_y;
	
	auto min_value_iterator = thrust::min_element(d_x.begin(), d_x.end());
	int min_x = min_value_iterator - d_x.begin();
	printf("Minimum X value at position: %d\n", min_x);
	
	auto max_value_iterator = thrust::max_element(d_y.begin(), d_y.end());
	int max_y = max_value_iterator - d_y.begin();
	printf("Maximum Y value at position: %d\n", max_y);
	
	// input points in device
    thrust::device_vector<Point> d_Points = host_Points;
    // vector of indexes of points belonging to the hull
    std::vector<int> hull_Points;

	
	return 0;
}
