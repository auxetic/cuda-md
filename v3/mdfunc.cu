#include "mdfunc.h"

#define BLOCK_SIZE_256  256
#define BLOCK_SIZE_512  512
#define BLOCK_SIZE_1024 1024

__managed__ double g_fmax;
__managed__ double g_wili;


__global__ void kernel_zero_confv( tpvec *thdconfv, int tnatom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < tnatom )
        {
        thdconfv[i].x = 0.0;
        thdconfv[i].y = 0.0;
        thdconfv[i].z = 0.0;
        }
    }

cudaError_t gpu_zero_confv( tpvec *thdconfv, tpbox tbox )
    {
    const int block_size = 256;
    const int natom = tbox.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_zero_confv <<< grids, threads >>> ( thdconfv, natom );

    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_update_vr( tpvec *thdcon, tpvec *thdconv, tpvec *thdconf, int tnatom, double dt )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < tnatom )
        {
        tpvec ra, va, fa;

        ra = thdcon[i];
        va = thdconv[i];
        fa = thdconf[i];

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

        thdconv[i] = va;
        thdcon[i]  = ra;
        }
    }

cudaError_t gpu_update_vr( tpvec *thdcon, tpvec *thdconv, tpvec *thdconf, tpbox tbox, double dt)
    {
    const int block_size = 256;

    const int natom = tbox.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_update_vr <<< grids, threads >>> ( thdcon, thdconv, thdconf, natom, dt );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_update_v( tpvec *thdconv, tpvec *thdconf, int tnatom, double hfdt )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < tnatom )
        {
        tpvec va, fa;
        va    = thdconv[i];
        fa    = thdconf[i];
        va.x += fa.x * hfdt;
        va.y += fa.y * hfdt;
        va.z += fa.z * hfdt;
        thdconv[i] = va;
        }
    }

cudaError_t gpu_update_v( tpvec *thdconv, tpvec *thdconf, tpbox tbox, double dt)
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;
    const double hfdt = 0.5 * dt;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_update_v <<< grids, threads >>> ( thdconv, thdconf, natom, hfdt );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


// calculate force between blocki and blockj
__global__ void kernel_calc_block_force( tpvec      *tdconf, 
                                           tponeblock *tdblocki, 
                                           tponeblock *tdblockj,  
                                           double     tlx )
    {
    __shared__ double sm_wili;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i >= tdblocki->natom )
        return;

    if ( threadIdx.x == 0 )
        sm_wili = 0.0;

    __syncthreads();

    //int nbsum = tonelist[i].nbsum;

    double xi = tdblocki->rx[i];
    double yi = tdblocki->ry[i];
    double zi = tdblocki->rz[i];
    double ri = tdblocki->radius[i];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    double wi = 0.0;

    int j;
    double xj, yj, zj, rj, rij, dij, Vr;
    for ( int jj=0; jj<tdblockj->natom; jj++ )
        {
        xj = tdblockj->rx[jj];
        yj = tdblockj->ry[jj];
        zj = tdblockj->rz[jj];
        rj = tdblockj->radius[jj];

        // xij and reuse xj, yj, zj for xij, xj is xij
        xj -= xi;
        yj -= yi;
        zj -= zi;
        xj -= round(xj/tlx)*tlx;
        yj -= round(yj/tlx)*tlx;
        zj -= round(zj/tlx)*tlx;

        rij = xj*xj + yj*yj + zj*zj;
        dij = ri + rj;

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

    tdconf[i].x = fx;
    tdconf[i].y = fy;
    tdconf[i].z = fz;

    atomicAdd( &sm_wili, wi );

    __syncthreads();
    if ( threadIdx.x == 0 )
        {
        //sm_wili /= (double ) sysdim;
        atomicAdd( &g_wili, sm_wili );
        }

    }

cudaError_t gpu_calc_force( tpvec *thdconf, 
                            tpblocks thdblocks, 
                            double *static_press, 
                            tpbox tbox )
    {
    const int block_size = 128;

    const int natom = tbox.natom;
    const double lx = tbox.len.x;

    const int nblocks = thdblocks.args.nblocks;
    const int nblockx = thdblocks.args.nblock.x;



    check_cuda( cudaDeviceSynchronize() );
    g_wili = 0.0;


    tponeblock *blocki, *blockj;
    int natomi;
    for ( int i = 0; i < nblocks; i++ )
        {
        // issue // debugger
        blocki = &thdblocks.oneblocks[i];
        natomi = blocki->natom;
        for( int jj = 0; jj < 25; jj++ )
            {
            j      = blocki->neighb[jj];
            blockj = &thdblocks.oneblocks[j];

            // debugger
            dim3 grids( (natomi/block_size)+1, 1, 1 );
            dim3 threads( block_size, 1, 1 );
            kernel_calc_block_force <<< grids, threads>>>( tpvec      *tdconf, 
                                                           tponeblock *tdblocki, 
                                                           tponeblock *tdblockj,  
                                                           double     tlx )
            }
        }

    check_cuda( cudaDeviceSynchronize() );

    *static_press = g_wili / (double) sysdim / pow(lx, sysdim);

    return cudaSuccess;
    }


__global__ void kernel_calc_fmax( tpvec *thdconf, int tnatom )
    {
    __shared__ double block_f[BLOCK_SIZE_256];
    const int tid = threadIdx.x;
    const int i   = threadIdx.x + blockIdx.x * blockDim.x;

    block_f[tid] = 0.0;

    if ( i < tnatom )
        block_f[tid] = fmax( fabs(thdconf[i].x), fabs(thdconf[i].y) );

    __syncthreads();

    int j = BLOCK_SIZE_256;
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

double gpu_calc_fmax( tpvec *thdconf, tpbox tbox )
    {
    const int block_size = BLOCK_SIZE_256;
    const int natom = tbox.natom;

    g_fmax = 0.0;

    dim3 grids( (natom/block_size)+1, 1, 1);
    dim3 threads( block_size, 1, 1);
    kernel_calc_fmax <<< grids, threads >>> ( thdconf, natom );
    check_cuda( cudaDeviceSynchronize() );

    return g_fmax;
    }
