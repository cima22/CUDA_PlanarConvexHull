#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <math.h>
#include "random_points.h"

// function that computes distance 
double distance(point p, point q) {
    double d = sqrt(pow((q.x - p.x),2) + pow((q.y - p.y),2));
	return d;
}

double distance_from_line(point t, point p, point q) {
    double a = p.y - q.y;
    double b = p.x - q.x;
    double c = p.y * (q.x - p.x) - p.x * (q.y - p.y);
    double d = fabs(a * p.x + b * p.y + c) / sqrt(pow(a,2) + pow(b,2));
    return d;
}
/* OLD SPLIT, NOT RECURSIVE
point split(point p, point q, point* points, int size) {
    point max_point = points[0]; // assume first point as baseline
    
    double max_dist = distance_from_line(p, q, points[0]);

    for (int i = 1; i < size; i++) {
        // check if point is above line by crossproduct
        double orientation = (q.y - p.y) * points[i].x - (q.x - p.x) * points[i].y + q.x * p.y - q.y * p.x;
		if (orientation > 0){
			// if it is above the line, then compute distance and compare it
			double dist = distance_from_line(p, q, points[i]);
        	if (dist > max_dist) {
            	max_point = points[i];
            	max_dist = dist;
        	}
        }
    }

    return max_point;
}
*/

void rsplit(point p, point q, point* points, int size, point* result, int* result_size) {

	if (*result_size >= size) {
		return;
	}
	
	point max_point = points[0]; // assume first point as baseline
    
    double max_dist = distance_from_line(p, q, points[0]);

    for (int i = 1; i < size; i++) {
        // check if point is above line by crossproduct
        double orientation = (q.y - p.y) * points[i].x - (q.x - p.x) * points[i].y + q.x * p.y - q.y * p.x;
		if (orientation > 0){
			// if it is above the line, then compute distance and compare it
			double dist = distance_from_line(p, q, points[i]);
        	if (dist > max_dist) {
            	max_point = points[i];
            	max_dist = dist;
        	}
        }
    }
    
    result[*result_size] = max_point;
    (*result_size)++;
    
    rsplit(p, max_point, points, size, result, result_size);
    rsplit(max_point, q, points, size, result, result_size);
}


int main() {
	 clock_t start = clock(); // save start time
	 
	struct point points[N];
	generate_random_points(points, N);
	
	printf("Points generated!\n");
	
	// array that will contain the points on the perimeter
	struct point chull[N];
	int hullsize = 0;
	
	// QuickHull logic starts here
	
	// example points
	point p = {-2353.12, -8840.96};
    point q = {9517.10, -6059.66};
    int size = sizeof(points) / sizeof(points[0]);

	//point max_point = split(p, q, points, size);
    rsplit(p, q, points, size, chull, &hullsize);
    rsplit(q, p, points, size, chull, &hullsize);

    //printf("Point with max distance from line pq: (%lf, %lf)\n", max_point.x, max_point.y);
	// QuickHull logic ends here
	
	clock_t end = clock();
	// compute elapsed time
	double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Execution time for Sequential QuickHull: %f seconds\n", elapsed_time);
	
	for(int i=0; i<20; i++) {
		printf("( x: %f , y: %f )\n", chull[i].x, chull[i].y);
	}

	return 0;
}