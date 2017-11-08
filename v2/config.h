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
extern tpvec  *con;
extern double *radius;
// device
extern tpvec  *dcon;
extern double *dradius;

// subroutines
void alloc_con( tpvec **_con, double **_radius, int _natom );
void gen_config( tpvec *_con, double *_radius, tpbox *_box, tpsets _sets );
void read_config( FILE *_fio, tpvec *_con, double *_radius, tpbox *_box );
void trim_config( tpvec *_con, tpbox _box );

cudaError_t device_alloc_con( tpvec **_con, double **_radius, int _natom );
cudaError_t gpu_trim_config( tpvec *_con, tpbox _box );

#endif
