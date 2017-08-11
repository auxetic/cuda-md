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

#define nlcut 0.9
#define listmax 24
#define maxn_of_block 256
#define mean_of_block 64

// type define
typedef struct
    {
    int   natom;
    float rx[maxn_of_block];
    float ry[maxn_of_block];
    float radius[maxn_of_block];
    int   tag[maxn_of_block];
    } tpblock;

typedef struct
    {
    int   nbsum;
    int   nb[listmax];
    float x, y;
    } tplist;

typedef struct
    {
    int    nblocks, nblockx, nblocky;
    double dlx, dly;
    } tpblockset;

// variables define
extern tpblockset hblockset;
extern tpblock *dblocks;
extern tplist  *dlist;

// subroutines
void calc_nblocks( tpblockset *tblockset, tpbox tbox );
void recalc_nblocks( tpblockset *tblockset, tpbox tbox );
cudaError_t gpu_make_hypercon( tpvec *tdcon, double *tdradius, tpbox tbox, tpblock *tdblocks, tpblockset tblockset );
cudaError_t gpu_make_list( tplist *tdlist, tpblock *tdblocks, tpvec *tdcon, tpblockset tblockset, tpbox tbox );
bool gpu_check_list( tpvec *tdcon, tpbox tbox, tplist *tdlist );

#endif
