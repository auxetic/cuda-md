#include "mdfunc.h"

__managed__ double g_fmax;
__managed__ double g_wili;

// calculate force of one block with all its neighbour at once
__global__ void kernel_calc_force_all_neighb_block(   vec_t      *tdconf, 
                                                      cell_t     *tdblocks, 
                                                      const int         tbid, 
                                                      const double      tlx )
    {
    __shared__ double sm_wili;

    __shared__ double rxi[max_size_of_block];
    __shared__ double ryi[max_size_of_block];
    __shared__ double rzi[max_size_of_block];
    __shared__ double ri [max_size_of_block];
    __shared__ double rxj[max_size_of_block];
    __shared__ double ryj[max_size_of_block];
    __shared__ double rzj[max_size_of_block];
    __shared__ double rj [max_size_of_block];

    const int i   = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;

    //if ( i >= tdblocki->natom )
    //    return;

    if ( tid == 0 ) sm_wili = 0.0;

    rxi[tid] = tdblocks[tbid].rx[i];
    ryi[tid] = tdblocks[tbid].ry[i];
    rzi[tid] = tdblocks[tbid].rz[i];
    ri [tid] = tdblocks[tbid].radius[i];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    double wi = 0.0;

    __syncthreads();

    int j;
    double rxij, ryij, rzij, rij, dij, Vr;
    cell_t *blocki, *blockj;
    blocki = &tdblocks[tbid];
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
        dij = ri[tid] + rj[j];
        
        if ( tid < blocki->natom && tid != j )
            {
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

        }
    // joint block force
    int bidj;
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
            dij = ri[tid] + rj[j];

            if ( tid < tdblocks[tbid].natom )
                {
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











cudaError_t gpu_calc_force( vec_t    *thdconf, 
                            hycon_t *thdblocks, 
                            double   *static_press, 
                            box_t tbox )
    {
    const int block_size = 128;

    const int natom = tbox.natom;
    const double lx = tbox.len.x;

    const int nblocks = thdblocks->args.nblocks;
    const int nblockx = thdblocks->args.nblock.x;

    check_cuda( cudaDeviceSynchronize() );
    g_wili = 0.0;


    cell_t *block;
    int grids, threads;
    for ( int i = 0; i < nblocks; i++ )
        {
        grids   = (nblocks/block_size)+1;
        threads = block_size;
        kernel_calc_force_all_neighb_block <<<grids, threads >>> ( thdconf, thdblocks->oneblocks, i, lx);
        }

    check_cuda( cudaDeviceSynchronize() );

    *static_press = g_wili / (double) sysdim / pow(lx, sysdim);

    return cudaSuccess;
    }




