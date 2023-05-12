# CUDA_PlanarConvexHull
Final project for the "GPU Architecures and Computing" @ TU Wien. <br>
Given a set of points P in R², compute the smallest subset V of P that spans the convex hull of P, i.e., the convex combination of points in V yields the smallest convex set that contains all points in P.

## Authors and acknowledgment

Project carried out by Cimador Gabriele, Eremia Andreea-Evelina, Stabile Marco and Tamborrino Michele.

## License

Repository licensed with the MIT license. See the [LICENSE](LICENSE) for rights and limitations.

## Description

random_point.h is the header where number of points and range are specified, as well as the point structure.\
\
points_generator.c contains the code that produces a set of points randomly in a plane, populating an array given as an argument.\
test.c is a file to test the effective points generation.\
quick_hull.c implements the sequential version of the quickhull algorithm.\
\
\
points_generator.cpp has a function that returns a vector of points of size defined in the .h file.\
main.cpp implements the sequential version of the quickhull algorithm.\
