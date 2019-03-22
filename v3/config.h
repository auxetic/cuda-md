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
#define max_size_of_cell 30
#define mean_size_of_cell 12

// block type to make up hyperconfiguration
typedef struct cell_t
    {
    int    natom;
    int    neighb[26];
    int    tag[max_size_of_cell];

    int    extraflag;
    struct cell_t *extra = NULL;

    double radius[max_size_of_cell];

    vec_t r[max_size_of_cell];
    vec_t v[max_size_of_cell];
    vec_t f[max_size_of_cell];

    } cell_t;

typedef struct
    {
    int    natom;
    int    nblocks;
    intv_t nblock;
    vec_t  blocklen;
    vec_t  dl;
    } hycon_args_t;

typedef struct
    {
    hycon_args_t args;
    cell_t      *blocks;
    } hycon_t;

// subroutines for configuration
void gen_config( vec_t *con, double *radius, box_t *box, sets_t sets );
void alloc_con( vec_t **con, double **radius, int natom );
void gen_lattice_fcc ( vec_t *con, double *radius, box_t *box, sets_t sets );
void read_config( FILE *tfio, vec_t *con, double *radius, box_t *box );
void write_config( FILE *tfio, vec_t *con, double *radius, box_t *box );
void trim_config( vec_t *con, box_t box );
void calc_boxl(double *radius, box_t *box);

cudaError_t alloc_managed_con( vec_t *con, double *radius, int natom );
cudaError_t gpu_trim_config( vec_t *con, box_t box );

void calc_nblocks( hycon_t *hycon, box_t box );
void recalc_nblocks( hycon_t *hycon, box_t box );

// subroutines for hyperconfiguration
void map( hycon_t hycon );
void calc_hypercon_args( hycon_t *hycon, box_t box );
void recalc_hypercon_args( hycon_t *hycon, box_t box );
cudaError_t alloc_managed_hypercon( hycon_t *hycon );
cudaError_t gpu_map_hypercon_con( hycon_t hycon, vec_t *, vec_t *, vec_t *, double *radius);
cudaError_t gpu_make_hypercon( hycon_t hycon, vec_t *con, double *radius, box_t box);

#endif
