#ifndef config_h
#define config_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "system.h"
#include "common.h"


// variables define
// host
extern vec_t  *con;
extern double *radius;
// device
extern vec_t  *dcon;
extern double *dradius;

// subroutines
void alloc_con( vec_t **tcon, double **tradius, int tnatom );
void gen_config( vec_t *tcon, double *tradius, tpbox *tbox, tpsets tsets );
void read_config( FILE *tfio, vec_t *tcon, double *tradius, tpbox *tbox );
void trim_config( vec_t *tcon, tpbox tbox );

cudaError_t device_alloc_con( vec_t **tcon, double **tradius, int tnatom );
cudaError_t gpu_trim_config( vec_t *tcon, tpbox tbox );

#endif
