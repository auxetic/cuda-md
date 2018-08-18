#include "md.h"

// internal function
__global__ void kernel_calc_chi( vec_t *thdconv, vec_t *thdconf, int natom );
cudaError_t gpu_calc_chi( vec_t *thdconv, vec_t *thdconf, box_t tbox, double *chi );
__global__ void kernel_modify_force( vec_t *thdconf, vec_t *thdconv, int natom, double tchi );
cudaError_t gpu_modify_force( vec_t *thdconf, vec_t *thdconv, box_t tbox, double tchi );

// internal variables
#define dt 0.01
vec_t  *hdcon, *hdconv, *hdconf;
double *hdradius;
tpblocks hdmdblock;
tplist   hdlist;

// kernel varialbes
#define BLOCK_SIZE_256  256
__managed__ double mpp, mpf;

tpmdset mdset;


void init_nvt( vec_t *thcon, double *thradius, box_t tbox, double ttemper )
    {
    // allocate config array
    cudaMallocManaged(&hdcon    , sizeof(vec_t)  * tbox.natom );
    cudaMallocManaged(&hdconv   , sizeof(vec_t)  * tbox.natom );
    cudaMallocManaged(&hdconf   , sizeof(vec_t)  * tbox.natom );
    cudaMallocManaged(&hdradius , sizeof(double) * tbox.natom );

    cudaMemcpy(hdcon,    thcon,    sizeof(vec_t)*tbox.natom,  cudaMemcpyHostToDevice);
    cudaMemcpy(hdradius, thradius, sizeof(double)*tbox.natom, cudaMemcpyHostToDevice);

    // list
    hdlist.natom = tbox.natom;
    cudaMallocManaged( &(hdlist.onelists), hdlist.natom*sizeof(tponelist) );
    calc_nblocks( &hdmdblock, tbox );
    cudaMallocManaged( &(hdmdblock.oneblocks), hdmdblock.args.nblocks*sizeof(tponeblock) );
    // make list
    printf( "making hypercon \n" );
    gpu_make_hypercon( hdmdblock, hdcon, hdradius, tbox );
    printf( "making list \n" );
    gpu_make_list( hdlist, hdmdblock, hdcon, tbox );

    mdset.temper = ttemper;
    }

void gpu_run_nvt( box_t tbox, double ttemper, int steps )
    {
    for ( int step=1; step <= steps; step++ )
        {
        // check and make list
        if ( gpu_check_list( hdlist, hdcon, tbox ) )
            {
            printf( "making list \n" );
            gpu_make_hypercon( hdmdblock, hdcon, hdradius, tbox );
            gpu_make_list( hdlist, hdmdblock, hdcon, tbox );
            }

        // velocity verlet / integrate veclocity and config
        gpu_update_vr( hdcon, hdconv, hdconf, tbox, dt );

        // temp
        double press;
        // calc force
        gpu_calc_force( hdconf, hdlist, hdcon, hdradius, &press, tbox );

        // nvt / modify force
        double chi;
        gpu_calc_chi( hdconv, hdconf, tbox, &chi );
        gpu_modify_force( hdconf, hdconv, tbox, chi );

        // velocity verlet / integrate velocity
        gpu_update_v( hdconv, hdconf, tbox, dt );
        }
    }

cudaError_t gpu_calc_chi( vec_t *thdconv, vec_t *thdconf, box_t tbox, double *chi )
    {
    const int block_size = 256;
    const int natom = tbox.natom;

    dim3 grids( ceil( natom / block_size )+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_calc_chi <<< grids, threads >>> ( thdconv, thdconf, natom );
    check_cuda( cudaDeviceSynchronize() );

    *chi = mpf/mpp;

    return cudaSuccess;
    }

__global__ void kernel_calc_chi( vec_t *thdconv, vec_t *thdconf, int natom )
    {
    __shared__ double spp[BLOCK_SIZE_256];
    __shared__ double spf[BLOCK_SIZE_256];

    const int tid = threadIdx.x;
    const int i   = tid + blockIdx.x * blockDim.x;

    spp[tid] = 0.0;
    spf[tid] = 0.0;

    if ( i < natom )
        {
        double fxi = thdconf[i].x;
        double fyi = thdconf[i].y;
        double vxi = thdconv[i].x;
        double vyi = thdconv[i].y;
        spp[tid] = vxi * vxi + vyi * vyi;
        spf[tid] = vxi * fxi + vyi * fyi;
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

cudaError_t gpu_modify_force( vec_t *thdconf, vec_t *thdconv, box_t tbox, double tchi )
    {
    const int block_size = 256;
    const int natom = tbox.natom;

    dim3 grids( ceil(natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_modify_force <<< grids, threads >>> ( thdconf, thdconv, natom, tchi );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

__global__ void kernel_modify_force( vec_t *thdconf, vec_t *thdconv, int natom, double tchi )
    {
    const int tid = threadIdx.x;
    const int i   = tid + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        thdconf[i].x = thdconf[i].x - tchi * thdconv[i].x;
        thdconf[i].y = thdconf[i].y - tchi * thdconv[i].y;
        }
    }
