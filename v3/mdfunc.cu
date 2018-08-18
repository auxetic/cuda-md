#include "mdfunc.h"

#define BLOCK_SIZE_256  256
#define BLOCK_SIZE_512  512
#define BLOCK_SIZE_1024 1024

__managed__ double g_fmax;
__managed__ double g_wili;


__global__ void kernel_zero_confv( vec_t *thdconfv, int tnatom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < tnatom )
        {
        thdconfv[i].x = 0.0;
        thdconfv[i].y = 0.0;
        }
    }

cudaError_t gpu_zero_confv( vec_t *thdconfv, box_t tbox )
    {
    const int block_size = 256;
    const int natom = tbox.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_zero_confv <<< grids, threads >>> ( thdconfv, natom );

    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_update_vr( vec_t *thdcon, vec_t *thdconv, vec_t *thdconf, int tnatom, double dt )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < tnatom )
        {
        vec_t ra, va, fa;

        ra = thdcon[i];
        va = thdconv[i];
        fa = thdconf[i];

        va.x += 0.5 * fa.x * dt;
        va.y += 0.5 * fa.y * dt;
        ra.x += va.x * dt;
        ra.y += va.y * dt;

        thdconv[i] = va;
        thdcon[i]  = ra;
        }
    }

cudaError_t gpu_update_vr( vec_t *thdcon, vec_t *thdconv, vec_t *thdconf, box_t tbox, double dt)
    {
    const int block_size = 256;

    const int natom = tbox.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_update_vr <<< grids, threads >>> ( thdcon, thdconv, thdconf, natom, dt );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_update_v( vec_t *thdconv, vec_t *thdconf, int tnatom, double hfdt )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < tnatom )
        {
        vec_t va, fa;
        va    = thdconv[i];
        fa    = thdconf[i];
        va.x += fa.x * hfdt;
        va.y += fa.y * hfdt;
        thdconv[i] = va;
        }
    }

cudaError_t gpu_update_v( vec_t *thdconv, vec_t *thdconf, box_t tbox, double dt)
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


__global__ void kernel_calc_force( vec_t *thdconf, tponelist *tonelist, vec_t *thdcon, double *thdradius, int tnatom, double tlx )
    {
    __shared__ double sm_wili;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i >= tnatom )
        return;

    if ( threadIdx.x == 0 )
        sm_wili = 0.0;

    __syncthreads();

    int nbsum = tonelist[i].nbsum;

    vec_t  rai = thdcon[i];
    vec_t  fai = { 0.0, 0.0 };
    double ri  = thdradius[i];
    double wi  = 0.0;

    for ( int jj=0; jj<nbsum; jj++ )
        {
        int j = tonelist[i].nb[jj];

        vec_t raj = thdcon[j];
        // dij equal to raidius of atom j
        double rj = thdradius[j];

        // xij
        raj.x -= rai.x;
        raj.y -= rai.y;
        raj.x -= round(raj.x/tlx)*tlx;
        raj.y -= round(raj.y/tlx)*tlx;

        double rij = raj.x*raj.x + raj.y*raj.y;
        double dij = ri + rj;

        if ( rij < dij*dij )
            {
            rij = sqrt(rij);

            double Vr = - ( 1.0 - rij/dij ) / dij;

            fai.x -= - Vr * raj.x / rij;
            fai.y -= - Vr * raj.y / rij;

            // wili
            wi += - Vr * rij;
            }
        }
    thdconf[i] = fai;

    atomicAdd( &sm_wili, wi );

    __syncthreads();
    if ( threadIdx.x == 0 )
        {
        sm_wili /= 2.0;
        atomicAdd( &g_wili, sm_wili );
        }

    }

cudaError_t gpu_calc_force( vec_t *thdconf, tplist thdlist, vec_t *thdcon, double *thdradius, double *static_press, box_t tbox )
    {
    const int block_size = 256;

    const int natom = tbox.natom;
    const double lx = tbox.len.x;

    g_wili = 0.0;
    check_cuda( cudaDeviceSynchronize() );

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_calc_force <<< grids, threads >>>( thdconf, thdlist.onelists, thdcon, thdradius, natom, lx );
    check_cuda( cudaDeviceSynchronize() );

    *static_press = g_wili / 2.0 / lx / lx;

    return cudaSuccess;
    }


__global__ void kernel_calc_fmax( vec_t *thdconf, int tnatom )
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

double gpu_calc_fmax( vec_t *thdconf, box_t tbox )
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
