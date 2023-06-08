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
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// compare two X points 
    struct PointComparatorX {
        __host__ __device__
        bool operator()(const Point& p1, const Point& p2) {
            return p1.x < p2.x;
        }
    };

struct PointComparator {
    __host__ __device__
    bool operator()(const Point& p1, const Point& p2) {
        if (p1.x < p2.x)
            return true;
        else if (p1.x > p2.x)
            return false;
        else
            return p1.y < p2.y;
    }
};


// functor to compute distance from line
struct DistanceFromLine : public thrust::unary_function<Point, double> {
    const Point left;
    const Point right;

    DistanceFromLine(const Point& left, const Point& right) : left(left), right(right) {}

    __host__ __device__
    double operator()(const Point& p) {
        double A = right.y - left.y;
        double B = left.x - right.x;
        double C = right.x * left.y - left.x * right.y;
        double dist = fabs(A * p.x + B * p.y + C) / sqrt(A * A + B * B);
        if (A * p.x + B * p.y + C < 0)
            dist = 0.0;
        return dist;
    }
};

struct isAboveClockwise : public thrust::unary_function<double, bool> {
    __host__ __device__
    bool operator()(double x) {
        return x > 0.0;
    }
};

void recursiveSplit(thrust::device_vector<Point>& output, const thrust::device_vector<Point>& input, int leftPoint, int rightPoint) {

	// check if output is already full
	if (output.size() >= N) return;

	const Point& left = input[leftPoint];
    const Point& right = input[rightPoint];

    // Calcola le distanze dei punti dalla retta
    thrust::device_vector<double> distances(input.size());
    thrust::transform(input.begin(), input.end(), distances.begin(), DistanceFromLine(left, right));

    // Trova il punto con la massima distanza dalla retta
    auto maxDistIter = thrust::max_element(distances.begin(), distances.end());
    size_t maxDistIndex = maxDistIter - distances.begin();
    
    // Aggiungi il punto trovato all'output
    output.push_back(input[maxDistIndex]);
	
	if (maxDistIndex != leftPoint) recursiveSplit(output, input, leftPoint, maxDistIndex);
	if (maxDistIndex != rightPoint) recursiveSplit(output, input, maxDistIndex, rightPoint);
}

void quickhullSplit(thrust::device_vector<Point>& output, const thrust::device_vector<Point>& input, int leftPoint, int rightPoint) {
    const Point& left = input[leftPoint];
    const Point& right = input[rightPoint];

    // Calcola le distanze dei punti dalla retta
    thrust::device_vector<double> distances(input.size());
    thrust::transform(input.begin(), input.end(), distances.begin(), DistanceFromLine(left, right));

    // Trova il punto con la massima distanza dalla retta
    auto maxDistIter = thrust::max_element(distances.begin(), distances.end());
    size_t maxDistIndex = maxDistIter - distances.begin();

    // Aggiungi il punto trovato all'output
    output.push_back(input[maxDistIndex]);
    
    if (maxDistIndex != leftPoint) recursiveSplit(output, input, leftPoint, maxDistIndex);
    
	if (maxDistIndex != rightPoint) recursiveSplit(output, input, maxDistIndex, rightPoint);
}




int main() {
	// create a vector of points in the plane with thrust
	thrust::host_vector<Point> h_points = generate_random_points();
	
	// create device vector with =
	thrust::device_vector<Point> d_points = h_points;
	
	// find point with minimum x (so the left point)
    auto min_x_iter = thrust::min_element(d_points.begin(), d_points.end(), PointComparatorX());
    size_t min_x_index = min_x_iter - d_points.begin();
    Point min_x_point = *min_x_iter;

    // find point with maximum x (so the right point)
    auto max_x_iter = thrust::max_element(d_points.begin(), d_points.end(), PointComparatorX());
    size_t max_x_index = max_x_iter - d_points.begin();
    Point max_x_point = *max_x_iter;
	
	thrust::device_vector<Point> d_out;
	
	quickhullSplit(d_out, d_points, min_x_index, max_x_index);
	quickhullSplit(d_out, d_points, max_x_index, min_x_index);
	
	// Ordina il vettore dei punti
	thrust::sort(d_out.begin(), d_out.end(), PointComparator());
	
	// Ridimensiona il vettore per eliminare i duplicati
	auto newEnd = thrust::unique(d_out.begin(), d_out.end(), PointComparator());
	d_out.resize(newEnd - d_out.begin());
	
	std::cout << "out size after unique: " << d_out.size() << std::endl;
	
	thrust::host_vector<Point> h_out = d_out;

    std::cout << "Convex Hull Points:" << std::endl;
    for (const auto& point : h_out)
    {
    	std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
    }
    std::cout << "Hull size: " << h_out.size() << std::endl;
  
	return 0;
}
