#ifndef config_h
#define config_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "system.h"


// type define
typedef struct
    {
    double x, y;
    } tpvec;

// variables define
// host
extern tpvec *con; 
extern double *radius;
// device
extern tpvec *dcon;
extern double *dradius;

// subroutines
void alloc_con( tpvec **tcon, double **tradius, int natom );
cudaError_t device_alloc_con( tpvec **tcon, double **tradius, int natom );
cudaError_t trans_con_to_gpu( tpvec *thcon, double *thradius, int natom, tpvec *tdcon, double *tdradius );
cudaError_t trans_con_to_host( tpvec *tdcon, double *tdradius, int natom, tpvec *thcon, double *thradius );
void gen_config( tpvec *tcon, double *tradius, tpbox *tbox, tpsets tsets );
void trim_config( tpvec *tcon, tpbox tbox );

#endif
