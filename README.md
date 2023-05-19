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
\
points_generator.cpp has a function that returns a vector of points of size defined in the .h file.\
serial_quickhull.cpp implements the sequential version of the quickhull algorithm.\
\
\
cuda_quickhull.cu is a first version of the QuickHull implemented with CUDA.
