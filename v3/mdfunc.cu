#include "mdfunc.h"

__managed__ double g_fmax;
__managed__ double g_wili;

// calculate force of one block with all its neighbour at once
__global__ void kernel_calc_force_all_neighb_block( hycon_t *hycon, 
                                                    const double lx )
    {
    const int bidi = blockIdx.x;
    const int tid  = threadIdx.x;

    __shared__ double         radi[max_size_of_cell];
    __shared__ double         radj[max_size_of_cell];
    __shared__ vec_t          ri[max_size_of_cell];
    __shared__ vec_t          rj[max_size_of_cell]; 
    __shared__ vec_t          f [max_size_of_cell];
    __shared__ double extern  wi[];

    //if ( tid == 0 ) blocki = blocks[bidi];
    const int natomi = hycon->cnatom[bidi];

    if ( tid < natomi ) 
        {
        ri[tid].x = hycon->r[bidi*msoc+tid].x;
        ri[tid].y = hycon->r[bidi*msoc+tid].y;
        ri[tid].z = hycon->r[bidi*msoc+tid].z;
        radi[tid] = hycon->radius[bidi*msoc+tid];
        }

    if ( tid < max_size_of_cell ) 
        {
        f[tid].x = 0.0;
        f[tid].y = 0.0;
        f[tid].z = 0.0;
        }

    if ( tid < (int) sqrt((double) blockDim.x) ) wi[tid] = 0.0;

    __syncthreads();

    const int i      = tid % natomi;
    const int j      = tid / natomi;

    // self block force
    if ( tid < natomi*natomi && i != j )
        {
        double rxij  = ri[j].x-ri[i].x;
        double ryij  = ri[j].y-ri[i].y;
        double rzij  = ri[j].z-ri[i].z;
        rxij -= round(rxij/lx)*lx;
        ryij -= round(ryij/lx)*lx;
        rzij -= round(rzij/lx)*lx;

        double rij = rxij*rxij + ryij*ryij + rzij*rzij;
        double dij = radi[j] + radi[i];
        
        if ( rij < dij*dij )
            {
            rij = sqrt(rij);

            double Vr = - ( 1.0 - rij/dij ) / dij;

            atomicAdd(&f[i].x, + Vr * rxij / rij);
            atomicAdd(&f[i].y, + Vr * ryij / rij);
            atomicAdd(&f[i].z, + Vr * rzij / rij);

            atomicAdd(&wi[i], - Vr * rij);
            printf("%26.16le\n", -Vr*rij);
            }
        }

    __syncthreads();

    // joint block force
    for ( int jj=0; jj<26; jj++ )
        {
        int bidj, natomj;

        bidj = hycon->neighb[26*bidi+jj];
        natomj = hycon->cnatom[bidj];

        if ( tid < natomj ) 
            {
            rj[tid].x = hycon->r[bidj*msoc+tid].x;
            rj[tid].y = hycon->r[bidj*msoc+tid].y;
            rj[tid].z = hycon->r[bidj*msoc+tid].z;
            radj[tid] = hycon->radius[bidj*msoc+tid];
            }
        __syncthreads();

        if ( tid < natomi*natomj )
            {
            double rxij  = rj[j].x-ri[i].x;
            double ryij  = rj[j].y-ri[i].y;
            double rzij  = rj[j].z-ri[i].z;
            rxij -= round(rxij/lx)*lx;
            ryij -= round(ryij/lx)*lx;
            rzij -= round(rzij/lx)*lx;

            double rij = rxij*rxij + ryij*ryij + rzij*rzij;
            double dij = radi[i] + radj[j];

            if ( rij < dij*dij )
                {
                rij = sqrt(rij);

                double Vr = - ( 1.0 - rij/dij ) / dij;

                atomicAdd(&f[i].x, + Vr * rxij / rij);
                atomicAdd(&f[i].y, + Vr * ryij / rij);
                atomicAdd(&f[i].z, + Vr * rzij / rij);

                atomicAdd(&wi[i], - Vr * rij);
                }
            }
        }

    int s = (int)sqrt((double)blockDim.x) / 2;
    while ( tid < s )
        {
        __syncthreads();
        wi[tid] += wi[tid+s];
        s >>= 1;
        }

    if ( tid < natomi )
        {
        hycon->f[bidi*msoc+tid].x = f[tid].x;
        hycon->f[bidi*msoc+tid].y = f[tid].y;
        hycon->f[bidi*msoc+tid].z = f[tid].z;
        }

    __syncthreads();

    if ( tid == 0 )
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

