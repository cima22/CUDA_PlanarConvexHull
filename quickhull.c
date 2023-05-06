#include <stdio.h>
#include <time.h> 
#include <math.h>
#include "random_points.h"

// function that computes distance 
double distance(point p, point q) {
    double d = sqrt((q.x - p.x)**2 + (q.y - p.y)**2)
	return d;
}

double distance_from_line(point t, point p, point q) {
    double a = p.y - q.y;
    double b = p.x - q.x;
    double c = p.y * (q.x - p.x) - p.x * (q.y - p.y);
    double d = fabs(a * px + b * py + c) / sqrt(a**2 + b**2);
    return d;
}

point split(point p, point q, point* points, int size) {
    point max_point = points[0]; // assume first point as baseline
    
    double max_dist = distance_from_line(p, q, points[0]);

    for (int i = 1; i < size; i++) {
        double dist = distance(p, q, points[i]);
        if (dist > max_dist) {
            max_point = points[i];
            max_dist = dist;
        }
    }

    return max_point;
}


int main() {
	 clock_t start = clock(); // save start time
	 
	struct point points[N];
	generate_random_points(points, N);
	
	printf("Points generated!\n");
	
	// QuickHull logic starts here
	
	// QuickHull logic ends here
	
	clock_t end = clock();
	// compute elapsed time
	double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Execution time for Sequential QuickHull: %f seconds\n", elapsed_time);

	return 0;
}