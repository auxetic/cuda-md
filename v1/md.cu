#include "md.h"

// internal function
__global__ void kernel_calc_chi( tpvec *tdconv, tpvec *tdconf, int natom );
cudaError_t gpu_calc_chi( tpvec *tdconv, tpvec *tdconf, tpbox tbox, double *chi );
__global__ void kernel_modify_force( tpvec *tdconf, tpvec *tdconv, int natom, double tchi );
cudaError_t gpu_modify_force( tpvec *tdconf, tpvec *tdconv, tpbox tbox, double tchi );

// internal variables
#define dt 0.01
tpvec  *dconv, *dconf;

// kernel varialbes
#define BLOCK_SIZE_256  256 
__managed__ double mpp, mpf;

tpmdset mdset;


void init_nvt( tpvec *thcon, double *thradius, tpbox tbox, double ttemper )
    {
    // allocate config array
    cudaMalloc((void **)&dcon    , sizeof(tpvec)  * tbox.natom );
    cudaMalloc((void **)&dconv   , sizeof(tpvec)  * tbox.natom );
    cudaMalloc((void **)&dconf   , sizeof(tpvec)  * tbox.natom );
    cudaMalloc((void **)&dradius , sizeof(double) * tbox.natom );

    cudaMemcpy(dcon,    thcon,    sizeof(tpvec)*tbox.natom,  cudaMemcpyHostToDevice);
    cudaMemcpy(dradius, thradius, sizeof(double)*tbox.natom, cudaMemcpyHostToDevice);

    mdset.temper = ttemper;
    }
    
void gpu_run_nvt( tpbox tbox, double ttemper, int steps )
    {
    for ( int step=1; step <= steps; step++ )
        {
        // check and make list
        if ( gpu_check_list( dcon, tbox, dlist ) )
            {
            printf( "making list \n" );
            gpu_make_hypercon( dcon, dradius, tbox, dblocks, hblockset );
            gpu_make_list( dlist, dblocks, dcon, hblockset, tbox );
            }

        // velocity verlet / integrate veclocity and config
        gpu_update_vr( dcon, dconv, dconf, tbox, dt );

        // temp
        double press;
        // calc force
        gpu_calc_force( dlist, dcon, dradius, dconf, &press, tbox );

        // nvt / modify force
        double chi;
        gpu_calc_chi( dconv, dconf, tbox, &chi );
        gpu_modify_force( dconf, dconv, tbox, chi );

        // velocity verlet / integrate velocity
        gpu_update_v( dconv, dconf, tbox, dt );
        }
    }

cudaError_t gpu_calc_chi( tpvec *tdconv, tpvec *tdconf, tpbox tbox, double *chi )
    {
    const int block_size = 256;
    const int natom = tbox.natom;

    dim3 grids( ceil( natom / block_size )+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_calc_chi <<< grids, threads >>> ( tdconv, tdconf, natom );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }
    
    *chi = mpf/mpp;

    return err;
    }

__global__ void kernel_calc_chi( tpvec *tdconv, tpvec *tdconf, int natom )
    {
    __shared__ double spp[BLOCK_SIZE_256];
    __shared__ double spf[BLOCK_SIZE_256];

    const int tid = threadIdx.x;
    const int i   = tid + blockIdx.x * blockDim.x;

    spp[tid] = 0.0;
    spf[tid] = 0.0;

    if ( i < natom )
        {
        double fxi = tdconf[i].x;
        double fyi = tdconf[i].y;
        double vxi = tdconv[i].x;
        double vyi = tdconv[i].y;
        spp[i] = vxi * vxi + vyi * vyi;
        spf[i] = vxi * fxi + vyi * fyi;
        }

    int j = blockDim.x;
    j >>= 1;
    while ( j != 0 )
        {
        if ( tid < j )
            {
            spp[tid] += spp[tid+j];
            spf[tid] += spf[tid+j];
            }
        __syncthreads();
        j >>= 1;
        }

    if ( tid == 0 )
        {
        atomicAdd( &mpp, spp[0] );
        atomicAdd( &mpf, spf[0] );
        }

    }

cudaError_t gpu_modify_force( tpvec *tdconf, tpvec *tdconv, tpbox tbox, double tchi )
    {
    const int block_size = 256;
    const int natom = tbox.natom;

    dim3 grids( ceil(natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    
    kernel_modify_force <<< grids, threads >>> ( tdconf, tdconv, natom, tchi );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }
    
    return err;
    }

__global__ void kernel_modify_force( tpvec *tdconf, tpvec *tdconv, int natom, double tchi )
    {
    const int tid = threadIdx.x;
    const int i   = tid + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        tdconf[i].x = tdconf[i].x - tchi * tdconv[i].x;
        tdconf[i].y = tdconf[i].y - tchi * tdconv[i].y;
        }

    }

