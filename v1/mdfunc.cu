#include "mdfunc.h"

#define BLOCK_SIZE_256  256 
#define BLOCK_SIZE_512  512 
#define BLOCK_SIZE_1024 1024 

__managed__ double mfmax;


__global__ void kernel_zero_confv( tpvec *tdconfv, int natom )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        tdconfv[i].y = 0.0;
        tdconfv[i].x = 0.0;
        }
    }

cudaError_t gpu_zero_confv( tpvec *tdconfv, tpbox tbox )
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;

    dim3 grids( ceil( natom/block_size ) + 1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_zero_confv <<< grids, threads >>> ( tdconfv, natom );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }


__global__ void kernel_update_vr( tpvec *tdcon, tpvec *tdconv, tpvec *tdconf, int natom, double dt, double hfdt )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        double xi, yi;
        xi = tdcon[i].x;
        yi = tdcon[i].y;

        double vxi, vyi;
        vxi = tdconv[i].x;
        vyi = tdconv[i].y;

        double fxi, fyi;
        fxi = tdconf[i].x;
        fyi = tdconf[i].y;

        vxi += fxi * hfdt;
        vyi += fyi * hfdt;

        xi  += vxi * dt;
        yi  += vyi * dt;

        tdcon[i].x = xi;
        tdcon[i].y = yi;

        tdconv[i].x = vxi;
        tdconv[i].y = vyi;
        }
    }

cudaError_t gpu_update_vr( tpvec *tdcon, tpvec *tdconv, tpvec *tdconf, tpbox tbox, double dt)
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;
    const double hfdt = 0.5 * dt;

    dim3 grids( ceil(natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_update_vr <<< grids, threads >>> ( tdcon, tdconv, tdconf, natom, dt, hfdt );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }


__global__ void kernel_update_v( tpvec *tdconv, tpvec *tdconf, int natom, double tdt, double hfdt )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        double vxi, vyi;
        vxi = tdconv[i].x;
        vyi = tdconv[i].y;

        double fxi, fyi;
        fxi = tdconf[i].x;
        fyi = tdconf[i].y;

        vxi += fxi * hfdt;
        vyi += fyi * hfdt;

        tdconv[i].x = vxi;
        tdconv[i].y = vyi;
        }
    }

cudaError_t gpu_update_v( tpvec *tdconv, tpvec *tdconf, tpbox tbox, double dt)
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;
    const double hfdt = 0.5 * dt;

    dim3 grids( ceil(natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_update_v <<< grids, threads >>> ( tdconv, tdconf, natom, dt, hfdt );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }

__global__ void kernel_calc_force( tplist *tdlist, tpvec *tdcon, double *tddradius, tpvec *tdconf, int natom, double lx, double ly )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i >= natom )
        return;

    int nbsum;
    nbsum = tdlist[i].nbsum;

    double fxi, fyi;
    fxi = 0.0;
    fyi = 0.0;

    double xi, yi, ri;
    xi = tdcon[i].x;
    yi = tdcon[i].y;
    ri = tddradius[i];

    for ( int jj=0; jj<nbsum; jj++ )
        {
        int j;
        j = tdlist[i].nb[jj];

        double xj, yj, rj;
        xj = tdcon[j].x;
        yj = tdcon[j].y;
        rj = tddradius[j];

        double xij, yij;
        xij = xj - xi;
        yij = yj - yi;
        xij = xij - round(xij/lx)*lx;
        yij = yij - round(yij/ly)*ly;

        double rij, dij;
        rij = xij*xij + yij*yij; // rij2
        dij = ri + rj;

        if ( rij < dij*dij )
            {
            rij = sqrt(rij);
            
            double fr; // wij
            fr = ( 1.0 - rij/dij ) / dij / rij;

            fxi -= fr * xij;
            fyi -= fr * yij;
            }
        }
    tdconf[i].x = fxi;
    tdconf[i].y = fyi;
    }

cudaError_t gpu_calc_force( tplist *tdlist, tpvec *tdcon, double *tddradius, tpvec *tdconf, tpbox tbox )
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;
    const double lx = tbox.x;
    const double ly = tbox.y;

    dim3 grids( ceil(natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_calc_force <<< grids, threads >>>( tdlist, tdcon, tddradius, tdconf, natom, lx, ly );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }

__global__ void kernel_calc_fmax( tpvec *tdconf, int natom )
    {
    __shared__ double block_f[BLOCK_SIZE_256];
    const int tid = threadIdx.x;
    const int i   = threadIdx.x + blockIdx.x * blockDim.x;

    block_f[tid] = 0.0;

    if ( i < natom )
        {
        block_f[tid] = fmax( abs(tdconf[i].x), abs(tdconf[i].y) );
        }

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
        {
        atomicMax( &mfmax, block_f[0] );
        }
    }

double gpu_calc_fmax( tpvec *tdconf, tpbox tbox )
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;

    dim3 grids( ceil(natom/block_size)+1, 1, 1);
    dim3 threads( block_size, 1, 1);

    kernel_calc_fmax <<< grids, threads >>> ( tdconf, natom );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return mfmax;
    }
