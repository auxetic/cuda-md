//>>vv main
#ifndef list_h
#define list_h

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "system.h"
#include "config.h"

#define nlcut         0.8e0
#define listmax       64
#define maxn_of_block 128
#define mean_of_block 64

// type define
typedef struct
    {
    int    natom;
    double rx[maxn_of_block];
    double ry[maxn_of_block];
    double radius[maxn_of_block];
    int    tag[maxn_of_block];
    } oneblock_t;

typedef struct
    {
    int    nblocks;
    intd   nblock;
    vec_t  dl;
    } blockargs_t;

typedef struct
    {
    blockargs_t args;
    oneblock_t  *oneblocks;
    } blocks_t;

typedef struct
    {
    int    nbsum;
    int    nb[listmax];
    double x, y;
    } onelist_t;

typedef struct
    {
    int natom;
    onelist_t *onelists;
    } list_t;

// variables define
extern blocks_t hdblocks;
extern list_t  *dlist;

// subroutines
void calc_nblocks( blocks_t *thdblocks, box_t tbox );
void recalc_nblocks( blocks_t *thdblocks, box_t tbox );
cudaError_t gpu_make_hypercon( blocks_t thdblocks, vec_t *tdcon, double *tdradius, box_t tbox );
cudaError_t gpu_make_list( list_t thdlist, blocks_t thdblocks, vec_t *tdcon, box_t tbox );
cudaError_t gpu_make_list_fallback( list_t thdlist, vec_t *tdcon, double *tradius, box_t tbox );
bool gpu_check_list( list_t thdlist, vec_t *tdcon, box_t tbox );
int cpu_make_list( list_t tlist, vec_t *tcon, double *tradius, box_t tbox );

#endif
