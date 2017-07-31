#include "mdfunc.h"

#define BLOCK_SIZE_256  256 
#define BLOCK_SIZE_512  512 
#define BLOCK_SIZE_1024 1024 

__managed__ double gsm_fmax;
__managed__ double gsm_wili;


__global__ void kernel_zero_confv( tpvec *tdconfv, int natom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
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

    dim3 grids( (natom/block_size)+1, 1, 1 );
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


__global__ void kernel_update_vr( tpvec *tdcon, tpvec *tdconv, tpvec *tdconf, int natom, double dt )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < natom )
        { 
        double r, v, f;

        r = tdcon[i].x;
        v = tdconv[i].x;
        f = tdconf[i].x;
        v += 0.5 * f * dt;
        r += v * dt;
        tdconv[i].x = v;
        tdcon[i].x += r;

        r = tdcon[i].y;
        v = tdconv[i].y;
        f = tdconf[i].y;
        v += 0.5 * f * dt;
        r += v * dt;
        tdconv[i].y = v;
        tdcon[i].y += r;
        }
    }

cudaError_t gpu_update_vr( tpvec *tdcon, tpvec *tdconv, tpvec *tdconf, tpbox tbox, double dt)
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_update_vr <<< grids, threads >>> ( tdcon, tdconv, tdconf, natom, dt );

    cudaError_t err;
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }


__global__ void kernel_update_v( tpvec *tdconv, tpvec *tdconf, int natom, double hfdt )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        double v, f;
        v  = tdconv[i].x;
        f  = tdconf[i].x;
        v += f * hfdt;
        tdconv[i].x = v;

        v  = tdconv[i].y;
        f  = tdconf[i].y;
        v += f * hfdt;
        tdconv[i].y = v;
        }
    }

cudaError_t gpu_update_v( tpvec *tdconv, tpvec *tdconf, tpbox tbox, double dt)
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;
    const double hfdt = 0.5 * dt;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_update_v <<< grids, threads >>> ( tdconv, tdconf, natom, hfdt );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }


__global__ void kernel_calc_force( tplist *tdlist, tpvec *tdcon, double *tdradius, tpvec *tdconf, int natom, double lx, double ly )
    {
    __shared__ double sm_wili;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i >= natom )
        return;

    if ( i == 0 ) sm_wili = 0.0;

    int nbsum = tdlist[i].nbsum;

    double fxi   = 0.0;
    double fyi   = 0.0;
    double wilii = 0.0;

    double xi = tdcon[i].x;
    double yi = tdcon[i].y;
    double ri = tdradius[i];

    for ( int jj=0; jj<nbsum; jj++ )
        {
        int j = tdlist[i].nb[jj];

        double xj  = tdcon[j].x;
        double yj  = tdcon[j].y;
        // dij equal to raidius of atom j
        double dij = tdradius[j];

        // xij
        xj -= xi;
        yj -= yi;
        xj -= round(xj/lx)*lx;
        yj -= round(yj/ly)*ly;

        double rij;
        rij = xj*xj + yj*yj; // rij2
        dij = ri + dij;

        if ( rij < dij*dij )
            {
            rij = sqrt(rij);
            
            //double fr  - > dij
            dij = ( 1.0 - rij/dij ) / dij / rij;

            // wili
            wilii += dij * rij*rij; 

            fxi -= dij * xj;
            fyi -= dij * yj;
            }
        }
    tdconf[i].x = fxi;
    tdconf[i].y = fyi;

    atomicAdd( &sm_wili, wilii );
    __syncthreads();
    atomicAdd( &gsm_wili, sm_wili );

    }

cudaError_t gpu_calc_force( tplist *tdlist, tpvec *tdcon, double *tdradius, tpvec *tdconf, double *static_press, tpbox tbox )
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;
    const double lx = tbox.x;
    const double ly = tbox.y;

    gsm_wili = 0.0;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_calc_force <<< grids, threads >>>( tdlist, tdcon, tdradius, tdconf, natom, lx, ly );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    *static_press = gsm_wili / 2.0 / 2.0 / lx / ly;

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
        atomicMax( &gsm_fmax, block_f[0] );
        }
    }

double gpu_calc_fmax( tpvec *tdconf, tpbox tbox )
    {
    const int block_size = BLOCK_SIZE_256;

    const int natom = tbox.natom;

    gsm_fmax = 0.0;

    dim3 grids( (natom/block_size)+1, 1, 1);
    dim3 threads( block_size, 1, 1);

    kernel_calc_fmax <<< grids, threads >>> ( tdconf, natom );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return gsm_fmax;
    }


