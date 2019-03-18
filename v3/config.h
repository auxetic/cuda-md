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
#define max_size_of_block 20
#define mean_size_of_block 12

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

    //cell_t *extrablock;
    } cell_t;

typedef struct
    {
    int    nblocks;
    intv_t nblock;
    vec_t  blocklen;
    vec_t  dl;
    } hyconargs_t;

typedef struct
    {
    hyconargs_t    args;
    cell_t    *oneblocks;
    } hycon_t;

// subroutines for configuration 
void gen_config( vec_t *tcon, double *tradius, box_t *tbox, sets_t tsets );
void alloc_con( vec_t **tcon, double **tradius, int tnatom );
void gen_lattice_fcc ( vec_t *tcon, double *tradius, box_t *tbox, sets_t tsets );
void read_config( FILE *tfio, vec_t *tcon, double *tradius, box_t *tbox );
void write_config( FILE *tfio, vec_t *tcon, double *tradius, box_t *tbox );
void trim_config( vec_t *tcon, box_t tbox );
void calc_boxl(double *tradius, box_t *tbox);

cudaError_t alloc_managed_con( vec_t *tcon, double *tradius, int tnatom );
cudaError_t gpu_trim_config( vec_t *tcon, box_t tbox );

void calc_nblocks( hycon_t *thdblocks, box_t tbox );
void recalc_nblocks( hycon_t *thdblocks, box_t tbox );

// subroutines for hyperconfiguration
void map( hycon_t thdblocks );
//void map( hycon_t *thdblocks );
void calc_hypercon_args( hycon_t *thdhycon, box_t tbox );
void recalc_hypercon_args( hycon_t *thdblock, box_t tbox );
cudaError_t alloc_managed_hypercon( hycon_t *thdhycon );
cudaError_t gpu_map_hypercon_con( hycon_t *thdblock, vec_t *thdcon, double *thdradius, box_t tbox);
cudaError_t gpu_make_hypercon( hycon_t thdblock, vec_t *thdcon, double *thdradius, box_t tbox);

#endif
