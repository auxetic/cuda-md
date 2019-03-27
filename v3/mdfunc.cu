#include "mdfunc.h"

__managed__ double g_fmax;
__managed__ double g_wili;

// calculate force of one block with all its neighbour at once
__global__ void kernel_calc_force_all_neighb_block( hycon_t *hycon, 
                                                    const double lx )
    {
    __shared__ double         radi[max_size_of_cell];
    __shared__ double         radj[max_size_of_cell];
    __shared__ vec_t          ri[max_size_of_cell];
    __shared__ vec_t          rj[max_size_of_cell]; 
    __shared__ vec_t          f [max_size_of_cell];
    __shared__ double extern  wi[];
    __shared__ int natomj;

    const unsigned short  natomi = hycon->cnatom[blockIdx.x];
    //if ( threadIdx.x == 0 ) natomi = hycon->cnatom[blockIdx.x];

    if ( threadIdx.x < natomi ) 
        {
        ri[threadIdx.x].x = hycon->r[blockIdx.x*msoc+threadIdx.x].x;
        ri[threadIdx.x].y = hycon->r[blockIdx.x*msoc+threadIdx.x].y;
        ri[threadIdx.x].z = hycon->r[blockIdx.x*msoc+threadIdx.x].z;
        radi[threadIdx.x] = hycon->radius[blockIdx.x*msoc+threadIdx.x];
        }

    if ( threadIdx.x < max_size_of_cell ) 
        {
        f[threadIdx.x].x = 0.0;
        f[threadIdx.x].y = 0.0;
        f[threadIdx.x].z = 0.0;
        }

    if ( threadIdx.x < (int) sqrt((double) blockDim.x) ) wi[threadIdx.x] = 0.0;

    __syncthreads();

    const unsigned short i      = threadIdx.x % natomi;
    const unsigned short j      = threadIdx.x / natomi;
    double rxij, ryij, rzij, rij, dij, Vr;

    // self block force
    if ( threadIdx.x < natomi*natomi && i != j )
        {
        rxij  = ri[j].x-ri[i].x;
        ryij  = ri[j].y-ri[i].y;
        rzij  = ri[j].z-ri[i].z;
        rxij -= round(rxij/lx)*lx;
        ryij -= round(ryij/lx)*lx;
        rzij -= round(rzij/lx)*lx;

        rij = rxij*rxij + ryij*ryij + rzij*rzij;
        dij = radi[j] + radi[i];
        
        if ( rij < dij*dij )
            {
            rij = sqrt(rij);

            Vr = - ( 1.0 - rij/dij ) / dij;

            atomicAdd(&f[i].x, + Vr * rxij / rij);
            atomicAdd(&f[i].y, + Vr * ryij / rij);
            atomicAdd(&f[i].z, + Vr * rzij / rij);

            atomicAdd(&wi[i], - Vr * rij);
            }
        }
    //__syncthreads();

    // joint block force
    for ( unsigned short jj=0; jj<26; jj++ )
        {
        unsigned short bidj;

        bidj = hycon->neighb[26*blockIdx.x+jj];
        if(threadIdx.x == 0) natomj = hycon->cnatom[bidj];

        if ( threadIdx.x < natomj ) 
            {
            rj[threadIdx.x].x = hycon->r[bidj*msoc+threadIdx.x].x;
            rj[threadIdx.x].y = hycon->r[bidj*msoc+threadIdx.x].y;
            rj[threadIdx.x].z = hycon->r[bidj*msoc+threadIdx.x].z;
            radj[threadIdx.x] = hycon->radius[bidj*msoc+threadIdx.x];
            }
        __syncthreads();

        if ( threadIdx.x < natomi*natomj )
            {
            rxij  = rj[j].x-ri[i].x;
            ryij  = rj[j].y-ri[i].y;
            rzij  = rj[j].z-ri[i].z;
            rxij -= round(rxij/lx)*lx;
            ryij -= round(ryij/lx)*lx;
            rzij -= round(rzij/lx)*lx;

            rij = rxij*rxij + ryij*ryij + rzij*rzij;
            dij = radi[i] + radj[j];

            if ( rij < dij*dij )
                {
                rij = sqrt(rij);

                Vr = - ( 1.0 - rij/dij ) / dij;

                //atomicAdd(&f[i].x, + Vr * rxij / rij);
                //atomicAdd(&f[i].y, + Vr * ryij / rij);
                //atomicAdd(&f[i].z, + Vr * rzij / rij);

                //atomicAdd(&wi[i], - Vr * rij);
                }
            }
        }

    unsigned short s = (int)sqrt((double)blockDim.x) / 2;
    while ( threadIdx.x < s )
        {
        __syncthreads();
        wi[threadIdx.x] += wi[threadIdx.x+s];
        s >>= 1;
        }

    if ( threadIdx.x < natomi )
        {
        hycon->f[blockIdx.x*msoc+threadIdx.x].x = f[threadIdx.x].x;
        hycon->f[blockIdx.x*msoc+threadIdx.x].y = f[threadIdx.x].y;
        hycon->f[blockIdx.x*msoc+threadIdx.x].z = f[threadIdx.x].z;
        }

    __syncthreads();

    if ( threadIdx.x == 0 )
        {
        atomicAdd(&g_wili, wi[0]);
        }

    }

cudaError_t gpu_calc_force( hycon_t *hycon, 
                            double  *press, 
                            box_t    box )
    {
    const int block_size = (int) pow( 2.0, ceil( log( (double) max_size_of_cell) / log(2.0) ) );

    const int nblocks = hycon->nblocks;

    const double lx = box.len.x;

    check_cuda( cudaDeviceSynchronize() );
    g_wili = 0.0;

    const int grids   = nblocks;
    const int threads = block_size * block_size;
    const int shared_mem_size = block_size*sizeof(double);
    printf("block size is %d x %d\n", block_size, nblocks);
    if ( threads > 1024 ) printf("#[WARNING] cell size too big\n");
    // should optimise to consider number of threads exceed maxisum size of a GPU block
    //kernel_calc_force_all_neighb_block <<<grids, threads>>> ( hycon, lx);
    kernel_calc_force_all_neighb_block <<<grids, threads, shared_mem_size >>> ( hycon, lx);

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

