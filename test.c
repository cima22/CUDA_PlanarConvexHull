#include <stdio.h>
#include <time.h> 
#include "random_points.h"


int main() {
	 clock_t start = clock(); // save start time
	 
	struct point points[N];
	generate_random_points(points, N);
	
	for (int i = 0; i < N; i++) {
		printf("Point %d: (%.2f, %.2f)\n", i+1, points[i].x, points[i].y);
	}
	
	clock_t end = clock();
	// compute elapsed time
	double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Execution time: %f seconds\n", elapsed_time);

	return 0;
}
