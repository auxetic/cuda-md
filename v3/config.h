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

// block type
typedef struct
    {
    int    natom;
    int    neighb[26];
    double rx[max_size_of_block];
    double ry[max_size_of_block];
    double rz[max_size_of_block];
    double radius[max_size_of_block];
    int    tag[max_size_of_block];

    tponeblock *extrablock;
    } tponeblock;

typedef struct
    {
    int    nblocks;
    intd   nblock;
    tpvec  blocklen;
    } tpblockargs;

typedef struct
    {
    int    natom;
    double rx[maxn_of_extra_block];
    double ry[maxn_of_extra_block];
    double rz[maxn_of_extra_block];
    double radius[maxn_of_extra_block];
    int    tag[maxn_of_extra_block];
    int    tagb[maxn_of_extra_block];
    } tpextrablock;

typedef struct
    {
    tpblockargs args;
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

// subroutines for block 
// set up map of block structure for use in calculating force // cpu 
void map( tpblocks thdblocks );
// find every block structure member // only gpu
cudaError_t gpu_make_hypercon();
// give the index of the block acording to the x,y,z integer  // cpu 
__inline__ int indexb( int ix, int iy, int iz, int m);
// what is the number of block this atom is
__devine__ __inline__ tpvec iblock(tpvec loc);
// calculate number of blocks of hypercon
void calc_nblocks( tpblocks *thdblocks, tpbox tbox );
// recaculate number of blocks of hypercon
void recalc_nblocks( tpblocks *thdblocks, tpbox tbox );

#endif
