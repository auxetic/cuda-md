#include "mdfunc.h"

__managed__ double g_fmax;
__managed__ double g_wili;


__global__ void kernel_zero_confv( vec_t *confv, int natom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < natom )
        {
        confv[i].x = 0.0;
        confv[i].y = 0.0;
        confv[i].z = 0.0;
        }
    }

cudaError_t gpu_zero_confv( vec_t *confv, box_t box )
    {
    const int block_size = 256;
    const int natom = box.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_zero_confv <<< grids, threads >>> ( confv, natom );

    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_update_vr( vec_t *con, vec_t *conv, vec_t *conf, int natom, double dt )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < natom )
        {
        vec_t ra, va, fa;

        ra = con[i];
        va = conv[i];
        fa = conf[i];

        /*
        va.x += 0.5 * fa.x * dt;
        va.y += 0.5 * fa.y * dt;
        va.z += 0.5 * fa.z * dt;
        ra.x += va.x * dt;
        ra.y += va.y * dt;
        ra.z += va.z * dt;
        */

        ra.x += va.x * dt + fa.x * dt * dt * 0.5;
        ra.y += va.y * dt + fa.y * dt * dt * 0.5;
        ra.z += va.z * dt + fa.z * dt * dt * 0.5;
        va.x += 0.5 * fa.x * dt;
        va.y += 0.5 * fa.y * dt;
        va.z += 0.5 * fa.z * dt;

        conv[i] = va;
        con[i]  = ra;
        }
    }

cudaError_t gpu_update_vr( vec_t *con, vec_t *conv, vec_t *conf, box_t box, double dt)
    {
    const int block_size = 256;

    const int natom = box.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_update_vr <<< grids, threads >>> ( con, conv, conf, natom, dt );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_update_v( vec_t *conv, vec_t *conf, int natom, double hfdt )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        vec_t va, fa;
        va    = conv[i];
        fa    = conf[i];
        va.x += fa.x * hfdt;
        va.y += fa.y * hfdt;
        va.z += fa.z * hfdt;
        conv[i] = va;
        }
    }

cudaError_t gpu_update_v( vec_t *conv, vec_t *conf, box_t box, double dt)
    {
    const int block_size = 256;

    const int natom = box.natom;
    const double hfdt = 0.5 * dt;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_update_v <<< grids, threads >>> ( conv, conf, natom, hfdt );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


// calculate force of one block with all its neighbour at once
__global__ void kernel_calc_force_all_neighb_block( vec_t        *conf, 
                                                    block_t      *tdblocks, 
                                                    const int    tbid, 
                                                    const double tlx )
    {
    __shared__ double sm_wili;

    __shared__ block_t block_core;
    __shared__ block_t block_edge;

    // __shared__ double rxi[max_size_of_block];
    // __shared__ double ryi[max_size_of_block];
    // __shared__ double rzi[max_size_of_block];
    // __shared__ double ri [max_size_of_block];
    // __shared__ double rxj[max_size_of_block];
    // __shared__ double ryj[max_size_of_block];
    // __shared__ double rzj[max_size_of_block];
    // __shared__ double rj [max_size_of_block];

    const int i   = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;

    if ( tid == 0 )
        sm_wili = 0.0;

    rxi[tid] = tdblocks[tbid]->rx[i];
    ryi[tid] = tdblocks[tbid]->ry[i];
    rzi[tid] = tdblocks[tbid]->rz[i];
    ri [tid] = tdblocks[tbid]->radius[i];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    double wi = 0.0;

    __syncthreads();

    int j;
    double rxij, ryij, rzij, rij, dij, Vr;
    //block_t *blocki, *blockj;
    blocki = tdblocks[tbid];
    // self block force
    rxj[tid] = rxi[tid];
    ryj[tid] = ryi[tid];
    rzj[tid] = rzi[tid];
    for ( int j=0; j<tdblocks[tbid].natom; j++)
        {
        rxij  =rxj[j]-rxi[tid];
        ryij  =ryj[j]-ryi[tid];
        rzij  =rzj[j]-rzi[tid];
        rxij -=round(rxij/tlx)*tlx;
        ryij -=round(ryij/tlx)*tlx;
        rzij -=round(rzij/tlx)*tlx;

        rij = rxij*rxij + ryij*ryij + rzij*rzij;
        dij = ri + rj;
        
        if ( tid < blocki->natom && tid != j )
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
    // joint block force
    for ( int jj=0; jj<26; jj++ )
        {
        bidj = tdblocks[tbid].neighb[jj];
        rxj[tid] = tdblocks[bidj].rx[tid];
        ryj[tid] = tdblocks[bidj].ry[tid];
        rzj[tid] = tdblocks[bidj].rz[tid];
        rj[tid]  = tdblocks[bidj].radius[tid];

        for ( int j = 0; j < tdblocks[bidj].natom; j++)
            {
            rxij  =rxj[j]-rxi[tid];
            ryij  =ryj[j]-ryi[tid];
            rzij  =rzj[j]-rzi[tid];
            rxij -=round(rxij/tlx)*tlx;
            ryij -=round(ryij/tlx)*tlx;
            rzij -=round(rzij/tlx)*tlx;

            rij = rxij*rxij + ryij*ryij + rzij*rzij;
            dij = ri + rj;

            if ( tid < tdblocks[tbid].natom )
                {
                if ( rij < dij*dij )
                    {
                    rij = sqrt(rij);

                    Vr = - ( 1.0 - rij/dij ) / dij;

                    fx -= - Vr * xj / rij;
                    fy -= - Vr * yj / rij;
                    fz -= - Vr * zj / rij;

                    // wili
                    wi += - Vr * rij;
                    }
                }
            }
        }

    conf[i].x = fx;
    conf[i].y = fy;
    conf[i].z = fz;

    atomicAdd( &sm_wili, wi );

    __syncthreads();
    if ( threadIdx.x == 0 )
        {
        //sm_wili /= (double ) sysdim;
        atomicAdd( &g_wili, sm_wili );
        }

    }







cudaError_t gpu_calc_force( vec_t    *conf, 
                            tpblocks *thdblocks, 
                            double   *static_press, 
                            box_t box )
    {
    const int block_size = 128;

    const int natom = box.natom;
    const double lx = box.len.x;

    const int nblocks = thdblocks.args.nblocks;
    const int nblockx = thdblocks.args.nblock.x;

    check_cuda( cudaDeviceSynchronize() );
    g_wili = 0.0;


    block_t *block;
    for ( int i = 0; i < nblocks; i++ )
        {
        grids   = (nblocks/block_size)+1;
        threads = block_size;
        kernel_calc_force_all_neighb_block <<<grids, threads >>> ( conf, thdblocks.oneblocks, i, lx);
        }

    check_cuda( cudaDeviceSynchronize() );

    *static_press = g_wili / (double) sysdim / pow(lx, sysdim);

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

    return g_fmax;
    }
