<img src="https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" /> <img src="https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white"/> 

# CUDA_PlanarConvexHull
Final project for the course [GPU Architecures and Computing](https://tiss.tuwien.ac.at/course/courseDetails.xhtml?courseNr=182731&semester=2023S&dswid=7054&dsrid=445) @ [TU Wien](https://www.tuwien.at/). <br>

## Convex hull
The *convex hull* of a set of points P in R² is the smallest subset V of P such that the convex combination of points in V yields the smallest convex set that contains all points in P.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/Animation_depicting_the_quickhull_algorithm.gif" alt="animated" /><br>
 Execution of the implemented algorithm used to compute the convex hull.
</p>

## Goal of the project
* Implement in C++ the [quickhull](https://en.wikipedia.org/wiki/Quickhull) algorithm, used to compute the convex hull of a set of points, using the software acceleration provided by the CUDA SDK.
* Test the implementations for correctness.
* Benchmark the different implementations in terms of scalability and efficiency compared to a full CPU implementation.

## Description
The project has been divided into independent modules to implement and compare:
1. A parametrized generator that can generate the required input for the algorithm.
2. A sequential version, which will serve as a baseline for comparison.
3. The parallel algorithm in CUDA using synchronous programming trying different memories:
   * Standard memory transfer
   * Pinned memory
   * Unified memory
   * Zero-copy memory
4. A version of the parallel algorithm in CUDA using the [Thrust](https://docs.nvidia.com/cuda/thrust/index.html) library.
5. Test the different implementations, by generating connected random sets of points and comparing the consistency of the results produced by the different implementations.
6. Perform an extensive performance analysis comparing the different implementations and the scalability of these algorithms with different numbers of points and dimensions.

## How to compile
A `Makefile` is provided in the main directory. Just type
```bash
make
```
To compile and build the executables, which will be present in the `bin` directory created by the `Makefile`.

## How to run
In the `bin` directory there will be present different executables:
* `serial_quickhull.x` which executes the serial implementation
* `cuda_quickhull.x` which executes the CUDA implementation with standard memory transfer
* `pinned_quickhull.x` which executes the CUDA implementation with pinned memory transfer
* `zero_quickhull.x` which executes the CUDA implementation with zero-copy memory transfer
* `unified_quickhull.x` which executes the CUDA implementation with the unified memory transfer
* `thrust_quickhull.x` which executes the CUDA implementation using the Thrust library

## Dependencies and prerequisites
To compile the code, it is necessary to have the [NVCC](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler) NVIDIA compiler installed in the system. Furthermore, to execute the code, it is necessary to have a CUDA compatible device installed in the system (you can check this [list](https://developer.nvidia.com/cuda-gpus)).

## Authors and acknowledgment
Project carried out by Cimador Gabriele, Eremia Andreea-Evelina, Stabile Marco and Tamborrino Michele.

## License
Repository licensed with the MIT license. See the [LICENSE](LICENSE) for rights and limitations.
