#include "cudamath.h"

// device subroutine

// overload atomicMax for double and float
__device__ double atomicMax(double* address, double val)
    {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
        {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong( val>__longlong_as_double(assumed) ? val : __longlong_as_double(assumed) )
        );
        } while (assumed != old);
    return __longlong_as_double(old);
    }

// overload atomicMax for double and float
__device__ float atomicMax(float* address, float val)
    {
    unsigned int* address_as_i = (unsigned int*) address;
    unsigned int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            //__float_as_int(fmaxf(val, __int_as_float(assumed))));
            __float_as_int( val>__int_as_float(assumed) ? val : __int_as_float(assumed) )
        );
    } while (assumed != old);
    return __int_as_float(old);
    }
