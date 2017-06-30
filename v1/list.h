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
//  int   round[listmax][sysdim];
    float x, y;
    } tplist;

typedef struct
    {
    int nblocks, nblockx, nblocky;
    double dblockx, dblocky;
    } tpblockset;

// variables define
extern __managed__ double mdrmax;
extern tpblock *hypercon;
extern tpblockset hyperconset;
extern tplist  *list;

// subroutines
void calc_nblocks( tpblockset *thyperconset, double lx, double ly, int natom );
cudaError_t gpu_make_hypercon( tpvec *tcon, double *tradius, tpbox tbox, tpblock *thypercon, tpblockset thyperconset );
cudaError_t gpu_make_list( tpblock *thypercon, tpvec *tcon, tpblockset thyperconset, tpbox tbox, tplist *tlist );
bool gpu_check_list( tpvec *tcon, tpbox tbox, tplist *tlist );

#endif
