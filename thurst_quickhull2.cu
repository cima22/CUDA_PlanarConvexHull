//
// Created by Gabriele on 08/06/2023.
//

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

struct PointComparatorByX {
    __host__ __device__
    bool operator()(const Point& p1, const Point& p2) {
        return p1.x < p2.x;
    }
};

struct isAboveLine{
private:
    const Point p;
    const Point q;
public:
    isAboveLine(const Point& p, const Point&q):p{p},q{q}{};

    __host__ __device__
    bool operator()(const Point& point) {
        return ((q.x - p.x) * (point.y - p.y)) - ((point.x - p.x) * (q.y - p.y)) > 0;
    }
};

struct DistanceFromLine {
private:
    double a;
    double b;
    double c;

public:
    DistanceFromLine(const Point& p, const Point& q) : a(q.y - p.y), b(p.x - q.x), c((q.x * p.y) - (p.x * q.y)){}

    __host__ __device__
    double operator()(const Point& p1, const Point& p2) {
        auto d1 = fabs((a * p1.x + b * p1.y + c) / std::sqrt(a * a + b * b));
        auto d2 = fabs((a * p2.x + b * p2.y + c) / std::sqrt(a * a + b * b));
        return d1 < d2;
    }
};

void quickHull(const thrust::device_vector<Point>& v, const Point& a, const Point& b, thrust::device_vector<Point>& hull) {
    if (v.empty()) {
        return;
    }

    Point f = *thrust::max_element(v.begin(),v.end(), DistanceFromLine(a,b));

    // Collect points to the left of segment (a, f)
    std::vector<Point> left;
    std::vector<Point> right;
    thrust::device_vector<Point> aboveAFsegment;
    thrust::copy_if(thurst::device,v.begin(),v.end(),aboveAFsegment.begin(),isAboveLine(a,f));
    thrust::device_vector<Point> aboveFBsegment;
    thrust::copy_if(thurst::device,v.begin(),v.end(),aboveFBsegment.begin(),isAboveLine(f,b));
    hull.push_back(f);
    quickHull(aboveAFsegment, a, f, hull);
    quickHull(aboveFBsegment, f, b, hull);

}

void quickHull(const thrust::device_vector<Point>& input, thrust::device_vector<Point>& output) {
    Point pointWithMinX = * thrust::min_element(input.begin(),input.end(),PointComparatorByX());
    Point pointWithMaxX = * thrust::max_element(input.begin(),input.end(),PointComparatorByX());
    thrust::device_vector<Point> aboveLine;
    thrust::copy_if(thurst::device,input.begin(),input.end(),aboveLine.begin(),isAboveLine(pointWithMinX,pointWithMaxX));
    thrust::device_vector<Point> belowLine;
    thrust::copy_if(thurst::device,input.begin(),input.end(),belowLine.begin(),isAboveLine(pointWithMaxX,pointWithMinX));
    output.push_back(pointWithMinX);
    quickHull(aboveLine, pointWithMinX, pointWithMaxX, output);
    output.push_back(pointWithMaxX);
    quickHull(belowLine, pointWithMinX, pointWithMaxX, output);
}

int main() {
    // create a vector of points in the plane with thrust
    thrust::host_vector<Point> h_points = generate_random_points();

    // create device vector
    thrust::device_vector<Point> d_points = h_points;
    thrust::device_vector<Point> hull;
    // find point with minimum x (so the left point)
    quickHull(d_points,hull);
    thrust::host_vector<Point> h_out = hull;

    std::cout << "Convex Hull Points:" << std::endl;
    for (const auto& point : h_out)
    {
        std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
    }
    std::cout << "Hull size: " << h_out.size() << std::endl;

    return 0;
}