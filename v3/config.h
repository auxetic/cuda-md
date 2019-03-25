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

// abbr
#define msoc max_size_of_cell

// block type to make up hyperconfiguration

typedef struct
    {

    bool   first_time;
    int    natom;
    int    nblocks;
    intv_t nblock;
    vec_t  dl;

    int    *extraflag;
    // dim(cnatom) = nblocks
    // cell_natom = cnatom[bid]
    int    *cnatom;
    // dim(neighb) = 26*nblocks
    // bidj = neighb[bid*26+jj]
    int    *neighb;

    // tag_of_atom = tag[bid*msoc+jj]
    int    *tag;
    // r = radius[bid*msoc+jj]
    double *radius;
    // dim(r) = msoc*sizeof(vec_t)*nblocks
    // ra.x = r[bid*msoc+jj].x
    vec_t *r;
    vec_t *v;
    vec_t *f;

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
void map( hycon_t *hycon );
void calc_hypercon_args( hycon_t *hycon, box_t box );
void recalc_hypercon_args( hycon_t *hycon, box_t box );
cudaError_t alloc_managed_hypercon( hycon_t *hycon );
cudaError_t gpu_map_hypercon_con( hycon_t *hycon, vec_t *, vec_t *, vec_t *, double *radius);
cudaError_t gpu_make_hypercon( hycon_t *hycon, vec_t *con, double *radius, box_t box);

#endif
