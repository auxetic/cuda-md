#ifndef cudamath_h
#define cudamath_h

#include <cuda.h>


__device__ double atomicMax(double* address, double val);

#endif
