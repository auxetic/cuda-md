#include "config.h"

// allocate memory space of config on device
cudaError_t alloc_managed_con( vec_t *tcon, double *tradius, int tnatom )
    {
    check_cuda( cudaMallocManaged( &tcon,     tnatom*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &tradius , tnatom*sizeof(double) ) );
    return cudaSuccess;
    }

// allocate memory space of hyperconfig as managed 
cudaError_t alloc_managed_hypercon( hycon_t *thdhycon )
    {
    int nblocks = thdhycon->args.nblocks;
    check_cuda( cudaMallocManaged( &thdhycon->oneblocks, nblocks*sizeof(cell_t) ) );
    return cudaSuccess;
    }

__global__ void kernel_trim_config( vec_t *tcon, int tnatom, double lx, double ly, double lz)
    {
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < tnatom )
        {
        double x, y, z;
        x = tcon[i].x;
        y = tcon[i].y;

        x -= round( x/lx ) * lx;
        y -= round( y/ly ) * ly;

        tcon[i].x = x;
        tcon[i].y = y;

#if sysdim == 3
        z = tcon[i].z;
        z -= round( z/lz ) * lz;
        tcon[i].z = z;
#endif
        }
    }

cudaError_t gpu_trim_config( vec_t *tcon, box_t tbox )
    {
    const int    natom = tbox.natom;
    const double lx    = tbox.len.x;
    const double ly    = tbox.len.y;
    const double lz    = tbox.len.z;

    const int    block_size = 256;
    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_trim_config<<< grids, threads >>>( tcon, natom, lx, ly, lz );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

// subroutines for hyperconfiguration

// calculate minimum image index of block with given rx, ry, rz
__inline__ __device__ int indexb( double rx, double ry, double rz, double tlx, int m)
    {
    double blocki = (double) m;
    
    rx /= tlx;
    ry /= tlx;
    rz /= tlx;
    rx -= round(rx);
    ry -= round(ry);
    rz -= round(rz);

    return  (int)floor((rx + 0.5) * blocki )     
           +(int)floor((ry + 0.5) * blocki ) * m
           +(int)floor((rz + 0.5) * blocki ) * m * m;
    }


__global__ void kernel_reset_hypercon_block(cell_t *block)
    {
    const int i = threadIdx.x;

    block->rx[i]     = 0.0;
    block->ry[i]     = 0.0;
    block->rz[i]     = 0.0;
    block->radius[i] = 0.0;
    block->tag[i]    = -1;
    }

// calculte index of each atom and register it into block structure // map config into hyperconfig
__global__ void kernel_make_hypercon( cell_t *tdoneblocks,
                                      vec_t      *tdcon,
                                      double     *tdradius,
                                      double      tlx,
                                      int         tnblockx,
                                      int         tnatom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= tnatom ) return;

    double rx = tdcon[i].x;
    double ry = tdcon[i].y;
    double rz = tdcon[i].z;
    double ri = tdradius[i];

    double x = rx;
    double y = ry;
    double z = rz;

    rx /= tlx;
    ry /= tlx;
    rz /= tlx;
    rx -= round(rx);
    ry -= round(ry);
    rz -= round(rz);

    int bid;
    bid =  (int) floor((rx + 0.5) * (double)tnblockx )                          
          +(int) floor((ry + 0.5) * (double)tnblockx ) * tnblockx             
          +(int) floor((rz + 0.5) * (double)tnblockx ) * tnblockx * tnblockx;

    int idxinblock = atomicAdd( &tdoneblocks[bid].natom, 1);
    if ( idxinblock < max_size_of_block - 2 )
        {
        tdoneblocks[bid].rx[idxinblock]     = x;
        tdoneblocks[bid].ry[idxinblock]     = y;
        tdoneblocks[bid].rz[idxinblock]     = z;
        tdoneblocks[bid].radius[idxinblock] = ri;
        tdoneblocks[bid].tag[idxinblock]    = i;
        }
    else
        {
        // TODO one should consider those exced the max number of atoms per blocks
        atomicSub( &tdoneblocks[bid].natom, 1 );
        //idxinblock = atomicAdd( &tdoneblocks[bid]->extrablock.natom, 1);
        //tdoneblocks[bid]->extrablock.rx[idxinblock]     = x;
        //tdoneblocks[bid]->extrablock.ry[idxinblock]     = y;
        //tdoneblocks[bid]->extrablock.rz[idxinblock]     = z;
        //tdoneblocks[bid]->extrablock.radius[idxinblock] = ri;
        //tdoneblocks[bid]->extrablock.tag[idxinblock]    = i;
        }
    }

cudaError_t gpu_make_hypercon( hycon_t *thdblock, vec_t *thdcon, double *thdradius, box_t tbox)
    {
    const int block_size = 256;
    const int nblocks    = thdblock->args.nblocks;
    const int nblockx    = thdblock->args.nblock.x;
    const int natom      = tbox.natom;
    const double lx      = tbox.len.x;

    int grids, threads;

    //reset hypercon
    cell_t *block;
    for ( int i = 0; i < nblocks; i++ )
        {
        block = &thdblock->oneblocks[i];

        block->natom = 0;
        threads = max_size_of_block;
        kernel_reset_hypercon_block <<<1, threads>>> (block);
        }
    check_cuda( cudaDeviceSynchronize() );

    // recalculate hypercon block length
    //recalc_hypercon_args(thdblock, tbox );

    // main
    grids   = (natom/block_size)+1;
    threads = block_size;

    kernel_make_hypercon <<< grids, threads >>>(thdblock->oneblocks,
                                                thdcon,
                                                thdradius,
                                                lx,
                                                nblockx,
                                                natom );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_map_hypercon_con (cell_t *tblock, vec_t *tdcon, double *tdradius)
    {
    const int tid = threadIdx.x;

    const int i   = tblock->tag[tid];

    if ( tid >= tblock->natom) return;

    tdcon[i].x = tblock->rx[tid];
    tdcon[i].y = tblock->ry[tid];
    tdcon[i].z = tblock->rz[tid];
    tdradius[i] = tblock->radius[tid];

    }
cudaError_t gpu_map_hypercon_con( hycon_t *thdblock, vec_t *thdcon, double *thdradius, box_t tbox)
    {
    const int nblocks    = thdblock->args.nblocks;
    //const int nblockx    = thdblock->args.nblock.x;
    //const int natom      = tbox.natom;
    //const double lx      = tbox.len.x;

    //map hypercon into normal con with index of atom unchanged
    int grids, threads;
    cell_t *block;
    for ( int i = 0; i < nblocks; i++ )
        {
        block = &thdblock->oneblocks[i];
        grids = 1;
        threads = max_size_of_block;
        kernel_map_hypercon_con<<<grids, threads>>>(block, thdcon, thdradius);
        }
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }
