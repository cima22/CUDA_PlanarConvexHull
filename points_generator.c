#include "random_points.h" // include header file

#include <stdlib.h>
#include <time.h>

void generate_random_points(struct point points[], int n) {
  // for debugging purposes, let's take a fixed seed for random numbers
  srand(2023);

  for (int i = 0; i < n; i++) {
    // genera le coordinate casuali per ogni punto
    points[i].x = ((double)rand() / RAND_MAX) * 2 * RANGE - RANGE;
    points[i].y = ((double)rand() / RAND_MAX) * 2 * RANGE - RANGE;
  }
}
