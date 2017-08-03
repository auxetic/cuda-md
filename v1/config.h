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
void alloc_con( tpvec **thcon, double **thradius, int tnatom );
cudaError_t device_alloc_con( tpvec **tdcon, double **tdradius, int tnatom );
cudaError_t trans_con_to_gpu(  tpvec *tdcon, double *tdradius, int tnatom, tpvec *thcon, double *thradius );
cudaError_t trans_con_to_host( tpvec *thcon, double *thradius, int tnatom, tpvec *tdcon, double *tdradius );
void gen_config( tpvec *thcon, double *thradius, tpbox *tbox, tpsets tsets );
void trim_config( tpvec *thcon, tpbox tbox );

#endif
