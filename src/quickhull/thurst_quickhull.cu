//
// Created by Gabriele on 08/06/2023.
//

#include <iostream>
#include <cmath>
#include "../../../../Desktop/CUDA_PlanarConvexHull/src/points_generation/random_points.h"
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
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

struct PointComparatorByX {
    __device__
    bool operator()(const Point& p1, const Point& p2) {
        return p1.x < p2.x;
    }
};

struct isAboveLine{
private:
    const Point p;
    const Point q;
public:
    __host__ __device__
    isAboveLine(const Point& p, const Point&q):p{p},q{q}{};

    __device__
    bool operator()(const Point& point) {
        return (((q.x - p.x) * (point.y - p.y)) - ((point.x - p.x) * (q.y - p.y))) > 1e-8;
    }
};

struct DistanceFromLine {
private:
    double a;
    double b;
    double c;

public:
    __host__ __device__
    DistanceFromLine(const Point& p, const Point& q) : a(q.y - p.y), b(p.x - q.x), c((q.x * p.y) - (p.x * q.y)){}

    __device__
    bool operator()(const Point& p1, const Point& p2) {
        auto d1 = fabs((a * p1.x + b * p1.y + c) / std::sqrt(a * a + b * b));
        auto d2 = fabs((a * p2.x + b * p2.y + c) / std::sqrt(a * a + b * b));
        return d1 < d2;
    }
};

__host__
void quickHull(const thrust::device_vector<Point>& v, const Point& a, const Point& b, thrust::device_vector<Point>& hull) {
    if (v.empty()) {
        return;
    }

    Point f = *thrust::max_element(v.begin(),v.end(), DistanceFromLine(a,b));

    thrust::device_vector<Point> aboveAFsegment(v.size());
    size_t aboveAFSize = thrust::copy_if(thrust::device,v.begin(),v.end(),aboveAFsegment.begin(),isAboveLine(a,f)) - aboveAFsegment.begin();
    aboveAFsegment.resize(aboveAFSize);
    thrust::device_vector<Point> aboveFBsegment(v.size());
    size_t aboveFBSize = thrust::copy_if(thrust::device,v.begin(),v.end(),aboveFBsegment.begin(),isAboveLine(f,b)) - aboveFBsegment.begin();
    aboveFBsegment.resize(aboveFBSize);
    hull.push_back(f);
    quickHull(aboveAFsegment, a, f, hull);
    quickHull(aboveFBsegment, f, b, hull);
}

__host__
void quickHull(const thrust::device_vector<Point>& input, thrust::device_vector<Point>& output) {
    Point pointWithMinX = * thrust::min_element(input.begin(),input.end(),PointComparatorByX());
    Point pointWithMaxX = * thrust::max_element(input.begin(),input.end(),PointComparatorByX());
    thrust::device_vector<Point> aboveLine(input.size());
    size_t aboveSize = thrust::copy_if(thrust::device,input.begin(),input.end(),aboveLine.begin(),isAboveLine(pointWithMinX,pointWithMaxX)) - aboveLine.begin();
    aboveLine.resize(aboveSize);
    thrust::device_vector<Point> belowLine(input.size());
    size_t belowSize = thrust::copy_if(thrust::device,input.begin(),input.end(),belowLine.begin(),isAboveLine(pointWithMaxX,pointWithMinX)) - belowLine.begin();
    belowLine.resize(belowSize);
    output.push_back(pointWithMinX);
    quickHull(aboveLine, pointWithMinX, pointWithMaxX, output);
    output.push_back(pointWithMaxX);
    quickHull(belowLine, pointWithMaxX, pointWithMinX, output);
}

int main() {
    // create a vector of points in the plane with thrust
    thrust::host_vector<Point> h_points = generate_random_points();
    std::cout << N << " points randomly generated!" << std::endl;

    // create device vector
    thrust::device_vector<Point> d_points = h_points;
    thrust::device_vector<Point> hull;

    
    auto start = std::chrono::high_resolution_clock::now();
    quickHull(d_points,hull);
    auto end = std::chrono::high_resolution_clock::now();
    thrust::host_vector<Point> h_out = hull;

    std::cout << "Found convex hull with " << h_out.size() << " points.\nPrinting the first and last three points of the hull:" << std::endl;

    // Print first 3 elements
    for (int i = 0; i < 3; i++) {
	    printf("(%f, %f)\n", h_out[i].x, h_out[i].y);
	}
	
    std::cout << ".\n.\n." << std::endl;
	
    // Print last 3 elements
    for (int i = h_out.size()-3; i < h_out.size(); i++) {
	    printf("(%f, %f)\n",h_out[i].x, h_out[i].y);
    }

    // Compute time interval
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << duration << " ms" << std::endl;
    return 0;
}
