#include "mdfunc.h"

__managed__ double g_fmax;
__managed__ double g_wili;

// calculate force of one block with all its neighbour at once
__global__ void kernel_calc_force_all_neighb_block( vec_t        *conf, 
                                                    cell_t       *blocks, 
                                                    //const int    bidi, 
                                                    const double lx )
    {
    const int bidi = blockIdx.x;
    const int tid  = threadIdx.x;

    __shared__ cell_t blocki, blockj;
    __shared__ double  fx[max_size_of_cell];
    __shared__ double  fy[max_size_of_cell];
    __shared__ double  fz[max_size_of_cell];
    __shared__ double  wi[64];

    if ( tid == 0 ) blocki = blocks[bidi];
    if ( tid < 64 ) wi[tid] = 0.0;
    if ( tid < max_size_of_cell ) 
        {
        fx[tid] = 0.0;
        fy[tid] = 0.0;
        fz[tid] = 0.0;
        }

    __syncthreads();

    const int i   = tid % blocki.natom;
    const int j   = tid / blocki.natom;

    // self block force
    if ( tid < blocki.natom*blocki.natom && i != j )
        {
        double rxij  = blocki.rx[j]-blocki.rx[i];
        double ryij  = blocki.ry[j]-blocki.ry[i];
        double rzij  = blocki.rz[j]-blocki.rz[i];
        rxij -= round(rxij/lx)*lx;
        ryij -= round(ryij/lx)*lx;
        rzij -= round(rzij/lx)*lx;

        double rij = rxij*rxij + ryij*ryij + rzij*rzij;
        double dij = blocki.radius[j] + blocki.radius[i];
        
        if ( rij < dij*dij )
            {
            rij = sqrt(rij);

            double Vr = - ( 1.0 - rij/dij ) / dij;

            atomicAdd(&fx[i], + Vr * rxij / rij);
            atomicAdd(&fy[i], + Vr * ryij / rij);
            atomicAdd(&fz[i], + Vr * rzij / rij);

            atomicAdd(&wi[i], - Vr * rij);
            }
        }

    // joint block force
    for ( int jj=0; jj<26; jj++ )
        {
        if ( tid == 0 )
            blockj = blocks[blocki.neighb[jj]];
        __syncthreads();

        if ( tid < blocki.natom*blockj.natom )
            {
            double rxij  = blockj.rx[j]-blocki.rx[i];
            double ryij  = blockj.ry[j]-blocki.ry[i];
            double rzij  = blockj.rz[j]-blocki.rz[i];
            rxij -= round(rxij/lx)*lx;
            ryij -= round(ryij/lx)*lx;
            rzij -= round(rzij/lx)*lx;

            double rij = rxij*rxij + ryij*ryij + rzij*rzij;
            double dij = blocki.radius[i] + blockj.radius[j];

            if ( rij < dij*dij )
                {
                rij = sqrt(rij);

                double Vr = - ( 1.0 - rij/dij ) / dij;

                atomicAdd(&fx[i], + Vr * rxij / rij);
                atomicAdd(&fy[i], + Vr * ryij / rij);
                atomicAdd(&fz[i], + Vr * rzij / rij);

                atomicAdd(&wi[i], - Vr * rij);
                }
            }
        }

    int s = 32 / 2;
    while ( tid < s )
        {
        __syncthreads();
        wi[tid] += wi[tid+s];
        s >>= 1;
        }

    if ( tid < blocki.natom )
        {
        int iatom = blocki.tag[tid];
        conf[iatom].x = fx[tid];
        conf[iatom].y = fy[tid];
        conf[iatom].z = fz[tid];
        }

    __syncthreads();

    if ( tid == 0 )
        {
        atomicAdd(&g_wili, wi[0]);
        }
    }

cudaError_t gpu_calc_force( vec_t   *conf, 
                            hycon_t *hycon, 
                            double  *press, 
                            box_t    box )
    {
    //const int block_size = (int) pow(2.0, ceil(sqrt((double)max_size_of_cell)));
    const int block_size = max_size_of_cell;

    const int nblocks = hycon->args.nblocks;

    const double lx = box.len.x;

    check_cuda( cudaDeviceSynchronize() );
    g_wili = 0.0;

    const int grids   = nblocks;
    const int threads = block_size * block_size;
    if ( threads > 1024 ) printf("#[WARNING] cell size too big\n");
    // should optimise to consider number of threads exceed maxisum size of a GPU block
    kernel_calc_force_all_neighb_block <<<grids, threads >>> ( conf, hycon->blocks, lx);

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

