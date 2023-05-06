#ifndef RANDOM_POINTS_H
#define RANDOM_POINTS_H

struct point {
  double x;
  double y;
};

void generate_random_points(struct point points[], int n);

#define N 20000
#define RANGE 30000

#endif // RANDOM_POINTS_H