#include "mdfunc.h"

__managed__ double g_fmax;
__managed__ double g_wili;

// calculate force of one block with all its neighbour at once
__global__ void kernel_calc_force_all_neighb_block( vec_t        *conf, 
                                                    block_t      *blocks, 
                                                    const int    bidi, 
                                                    const double lx,
                                                    const int    block_size)
    {
    __shared__ double sm_wili;
    __shared__ block_t blocki, blockj;

    __shared__ double fx[max_size_of_block];
    __shared__ double fy[max_size_of_block];
    __shared__ double fz[max_size_of_block];
    __shared__ double wi[max_size_of_block];

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int i   = threadIdx.x / block_size;
    const int j   = threadIdx.x % block_size;

    if ( i == 0 ) sm_wili = 0.0;
    if ( i == 1 ) blocki = blocks[bidi];
    if ()

    __syncthreads();

    double rxij, ryij, rzij, rij, dij, Vr;
    // self block force
    if ( i < blocki.natom )
        {
        for ( int j=0; j<blocki.natom; j++)
            {
            rxij  = blocki.rx[j]-blocki.rx[i];
            ryij  = blocki.ry[j]-blocki.ry[i];
            rzij  = blocki.rz[j]-blocki.rz[i];
            rxij -= round(rxij/lx)*lx;
            ryij -= round(ryij/lx)*lx;
            rzij -= round(rzij/lx)*lx;

            rij = rxij*rxij + ryij*ryij + rzij*rzij;
            dij = blocki.radius[j] + blocki.radius[i];
            
            if ( rij < dij*dij && i != j)
                {
                rij = sqrt(rij);

                Vr = - ( 1.0 - rij/dij ) / dij;

                fx -= - Vr * rxij / rij;
                fy -= - Vr * ryij / rij;
                fz -= - Vr * rzij / rij;

                // wili
                wi += - Vr * rij;
                }
            }
        }

    // joint block force
    for ( int jj=0; jj<26; jj++ )
        {
        if ( i == blockDim.x - 1 )
            blockj = blocks[blocki.neighb[jj]];

        __syncthreads();

        if ( i < blocki.natom )
            {
            for ( int j = 0; j < blockj.natom; j++)
                {
                rxij  = blockj.rx[j]-blocki.rx[i];
                ryij  = blockj.ry[j]-blocki.ry[i];
                rzij  = blockj.rz[j]-blocki.rz[i];
                rxij -= round(rxij/lx)*lx;
                ryij -= round(ryij/lx)*lx;
                rzij -= round(rzij/lx)*lx;

                rij = rxij*rxij + ryij*ryij + rzij*rzij;
                dij = blocki.radius[i] + blockj.radius[j];

                if ( rij < dij*dij )
                    {
                    rij = sqrt(rij);

                    Vr = - ( 1.0 - rij/dij ) / dij;

                    fx -= - Vr * rxij / rij;
                    fy -= - Vr * ryij / rij;
                    fz -= - Vr * rzij / rij;

                    // wili
                    wi += - Vr * rij;
                    }
                }
            }
        }

    int iatom;
    if ( i < blocki.natom )
        {
        iatom = blocki.tag[i];
        conf[iatom].x = fx;
        conf[iatom].y = fy;
        conf[iatom].z = fz;

        atomicAdd( &sm_wili, wi );
        }

    __syncthreads();

    if ( i == 3 )
        {
        atomicAdd( &g_wili, sm_wili );
        }
    }

cudaError_t gpu_calc_force( vec_t   *conf, 
                            hycon_t *hycon, 
                            double  *press, 
                            box_t    box )
    {
    const int block_size = (int) pow(2.0, ceil(sqrt((double)max_size_of_cell)));

    const int natom = box.natom;
    const double lx = box.len.x;

    const int nblocks = hycon->args.nblocks;
    const int nblockx = hycon->args.nblock.x;

    check_cuda( cudaDeviceSynchronize() );
    g_wili = 0.0;

    int grids, threads;
    printf("desine block_size is %d x %d\n", block_size, nblocks);
    grids   = 1;
    threads = block_size * block_size;
    if ( threads > 1024 ) 
        {
        printf("cell size too big\n");
        }
    for ( int i = 0; i < nblocks; i++ )
        {
        // should optimise to consider number of threads exceed maxisum size of a GPU block
        kernel_calc_force_all_neighb_block <<<grids, threads >>> ( conf, hycon->blocks, i, lx, block_size);
        }

    check_cuda( cudaDeviceSynchronize() );

    *press = g_wili / (double) sysdim / pow(lx, sysdim);

    return cudaSuccess;
    }

__global__ void kernel_calc_fmax( vec_t *conf, int natom )
    {
    __shared__ double block_f[256];
    const int tid = threadIdx.x;
    const int i   = threadIdx.x + blockIdx.x * blockDim.x;

    block_f[tid] = 0.0;

    if ( i < natom )
        block_f[tid] = fmax( fabs(conf[i].x), fabs(conf[i].y) );

    __syncthreads();

    int j = 256;
    j >>= 1;
    while ( j != 0 )
        {
        if ( tid < j )
            {
            block_f[tid] = fmax( block_f[tid], block_f[tid+j] );
            }
        __syncthreads();
        j >>= 1;
        }

    if ( tid == 0 )
        atomicMax( &g_fmax, block_f[0] );
    }

double gpu_calc_fmax( vec_t *conf, box_t box )
    {
    const int block_size = 256;
    const int natom = box.natom;

    g_fmax = 0.0;

    dim3 grids( (natom/block_size)+1, 1, 1);
    dim3 threads( block_size, 1, 1);
    kernel_calc_fmax <<< grids, threads >>> ( conf, natom );
    check_cuda( cudaDeviceSynchronize() );

    }

