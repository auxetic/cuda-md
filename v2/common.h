#ifndef __common_h__
#define __common_h__

#include <stdio.h>

static void check_cuda_error( cudaError_t err, const char *file, int line ) 
    {
    if (err != cudaSuccess) 
        {
        fprintf( stderr, "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit(-1);
        }
    }

#define check_cuda( err ) ( check_cuda_error( err, __FILE__, __LINE__ ) )

#endif
