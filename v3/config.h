#ifndef config_h
#define config_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "system.h"
#include "common.h"

// max size of a block
#define max_size_of_block 128
#define mean_size_of_block 64

// global variables define
// host
extern vec_t  *con;
extern double *radius;
// device
extern vec_t  *dcon;
extern double *dradius;

// block type to make up hyperconfiguration
typedef struct
    {
    int    natom;
    int    neighb[26];

    int    tag[max_size_of_block];
    double rx[max_size_of_block];
    double ry[max_size_of_block];
    double rz[max_size_of_block];
    double radius[max_size_of_block];

    tponeblock *extrablock;
    } tponeblock;

typedef struct
    {
    int    nblocks;
    intv_t nblock;
    vec_t  blocklen;
    } tpblockargs;

typedef struct
    {
    tpblockargs    args;
    tponeblock    *oneblocks;
    } tpblocks;

// subroutines for configuration 
void alloc_con( tpvec **tcon, double **tradius, int tnatom );
void gen_config( tpvec *tcon, double *tradius, tpbox *tbox, tpsets tsets );
void gen_lattice_fcc ( tpvec *tcon, double *tradius, tpbox *tbox, tpsets tsets );
void read_config( FILE *tfio, tpvec *tcon, double *tradius, tpbox *tbox );
void write_config( FILE *tfio, tpvec *tcon, double *tradius, tpbox *tbox );
void trim_config( tpvec *tcon, tpbox tbox );
void calc_boxl(double *tradius, tpbox *tbox);

cudaError_t device_alloc_con( tpvec **tcon, double **tradius, int tnatom );
cudaError_t gpu_trim_config( tpvec *tcon, tpbox tbox );

void calc_nblocks( tpblocks *thdblocks, tpbox tbox );
void recalc_nblocks( tpblocks *thdblocks, tpbox tbox );

// subroutines for hyperconfiguration
cudaError_t gpu_make_hypercon();

#endif
