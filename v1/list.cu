#include "list.h"

// variables define
__managed__ double mdrmax;
__managed__ int need_remake;
tpblock *hypercon;
tpblockset hyperconset;
tplist  *list;

// device subroutine
__device__ double atomicMax_list(double* address, double val)
    {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do 
        {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong( val>__longlong_as_double(assumed) ? val : __longlong_as_double(assumed) ) 
        );
        } while (assumed != old);
    return __longlong_as_double(old);
    }

void calc_nblocks( tpblockset *thyperconset, double lx, double ly, int natom )
    {
    int nblockx = ceil( sqrt( (double)natom / mean_of_block ) );
    int nblocky = nblockx;

    int nblocks = nblockx * nblocky;

    double dlx = lx / nblockx;
    double dly = ly / nblockx;

    thyperconset->nblocks = nblocks;
    thyperconset->nblockx = nblockx;
    thyperconset->nblocky = nblocky;
    thyperconset->dblockx = dlx;
    thyperconset->dblocky = dly;
    }

__global__ void kernel_make_hypercon( tpvec *tcon, double *tradius, 
                           int natom,
                           double lx, double ly, 
                           double dblockx, double dblocky, 
                           int nblockx,
                           tpblock *thypercon )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= natom )
        return;

    float xi, yi, ri;
    xi = tcon[i].x;
    yi = tcon[i].y;
    ri = tradius[i];

    float xi0, yi0;
    xi0 = xi - round( xi / lx ) * lx;
    yi0 = yi - round( yi / ly ) * ly;

    int bidx, bidy, bid;
    bidx = floor( (xi0+0.5*lx) / dblockx );
    bidy = floor( (yi0+0.5*ly) / dblocky );

    bid = bidx + bidy * nblockx;

    int idxtemp = atomicAdd( &thypercon[bid].natom, 1 );

    if ( idxtemp < maxn_of_block )
        {
        thypercon[bid].rx[idxtemp]     = xi;
        thypercon[bid].ry[idxtemp]     = yi;
        thypercon[bid].radius[idxtemp] = ri;
        thypercon[bid].tag[idxtemp]    = i;
        }

    if ( idxtemp == ( maxn_of_block-2 ) )
        return;
    }

__global__ void kernel_init_hypercon( tpblock *thypercon, int tnblocks )
    {
    const int i   = threadIdx.x + blockDim.x * blockIdx.x;

    if ( i < tnblocks )
        {
        thypercon[i].natom = 0;
        }
    }

cudaError_t gpu_make_hypercon( tpvec *tcon, double *tradius, tpbox tbox, tpblock *thypercon, tpblockset thyperconset )
    {
    int  block_size = 256;

    // set hypercon.natom to zero
    kernel_init_hypercon <<< ceil(thyperconset.nblocks/block_size)+1, block_size >>> ( thypercon, thyperconset.nblocks );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    // main
    dim3 grids(ceil(tbox.natom/block_size)+1,1,1);
    dim3 threads(block_size,1,1);

    kernel_make_hypercon <<< grids, threads >>>( tcon, tradius, 
                          tbox.natom, 
                          tbox.x, tbox.y, 
                          thyperconset.dblockx, thyperconset.dblocky,
                          thyperconset.nblockx,
                          thypercon );

    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }

__global__ void kernel_make_list( tpblock *thypercon, float lx, float ly, tplist *tlist )
    {
    __shared__ tpblock blocks[9];

    int bidx = blockIdx.x;  
    int bidy = blockIdx.y;
    int bid  = bidx + bidy * gridDim.x;
    int atomi = threadIdx.x;

    if ( atomi < 9 ) 
        {
        int i, j;
        i = atomi%3-1;
        j = atomi/3-1;

        int joffset;
        joffset = bid + i + gridDim.x * j;

        // most left or right column
        if ( bidx == 0           && i==-1 ) joffset += gridDim.x;
        if ( bidy == 0           && j==-1 ) joffset += gridDim.x*gridDim.y;
        if ( bidx == gridDim.x-1 && i==1  ) joffset -= gridDim.x;
        if ( bidy == gridDim.y-1 && j==1  ) joffset -= gridDim.x*gridDim.y;

        blocks[atomi] = thypercon[joffset];
        }

    __syncthreads();

    if ( atomi < blocks[4].natom ) 
        {
        float xi, yi, ri;
        int itag;
        // get xyr from shared memory
        xi = blocks[4].rx[atomi];
        yi = blocks[4].ry[atomi];
        ri = blocks[4].radius[atomi];
        itag = blocks[4].tag[atomi];
        // save xy to tlist
      //tlist[itag].x = xi;
      //tlist[itag].y = yi;
      //tlist[itag].nbsum = 0;
        for ( int ib=0; ib<9; ib++ ) 
            {
            for ( int atomj=0; atomj<blocks[ib].natom; atomj++ ) 
                {
                float xj, yj, rj;
                int jtag;
                xj = blocks[ib].rx[atomj];
                yj = blocks[ib].ry[atomj];
                rj = blocks[ib].radius[atomj];
                jtag = blocks[ib].tag[atomj];
                if ( itag == jtag ) continue;
                float xij, yij, rij2, dijcut2;
                xij = xj - xi;
                yij = yj - yi;
                dijcut2 = ri + rj + nlcut;
                dijcut2 = dijcut2 * dijcut2;
                int iroundx, iroundy;
                iroundx = (int)round(xij/lx);
                iroundy = (int)round(yij/ly);

                xij = xij - iroundx * lx;
                yij = yij - iroundy * ly;

                rij2 = xij*xij + yij*yij;

                if ( rij2 < dijcut2 ) 
                    {
                    if ( tlist[itag].nbsum == listmax - 2 ) continue;
                    tlist[itag].nbsum += 1;
                    tlist[itag].nb[tlist[itag].nbsum-1] = jtag;
//                  tlist[itag].round[tlist[itag].nbsum-1][0] = iroundx;
//                  tlist[itag].round[tlist[itag].nbsum-1][1] = iroundy;
                    }
                }
            }
        }
    }

__global__ void kernel_zero_list( tpvec *tcon, tplist *tlist, int natom )
    {
    const int tid = threadIdx.x;
    const int i   = tid + blockDim.x * blockIdx.x;

    if ( i < natom )
        {
        tlist[i].nbsum = 0;
        tlist[i].x = tcon[i].x;
        tlist[i].y = tcon[i].y;
        }
    }

cudaError_t gpu_make_list( tpblock *thypercon, tpvec *tcon, tpblockset thyperconset, tpbox tbox, tplist *tlist )
    {
    const int block_size = 256;
    float lx = tbox.x;
    float ly = tbox.y;

    kernel_zero_list <<< ceil(tbox.natom/block_size)+1, block_size >>> ( tcon, tlist, tbox.natom );
    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    dim3 grids(thyperconset.nblockx,thyperconset.nblocky,1);
    dim3 threads(maxn_of_block,1,1);
    kernel_make_list <<< grids, threads >>> ( thypercon, lx, ly, tlist );

    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }


__global__ void kernel_check_list( tpvec *tcon, int natom, tplist *tlist )
    {
    __shared__ int block_need_remake;
    __shared__ double block_drmax;

    const int i   = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;

    if ( i >= natom )
        return;

    if ( tid == 0 ) 
        {
        block_need_remake = 0;
        block_drmax = 0.0;
        }

    __syncthreads();

    float xi, yi;
    xi = tcon[i].x;
    yi = tcon[i].y;

    double th_max_dis, dr1, dr2; //dr3;
    dr1 = xi - tlist[i].x;
    dr2 = yi - tlist[i].y;

    th_max_dis = 2.0 * 1.42 * fmax(dr1, dr2);

    atomicMax_list( &block_drmax, th_max_dis );

    if ( th_max_dis > nlcut ) 
        block_need_remake = 1;

    __syncthreads();
    
    if ( tid == 0 )
        {
        atomicMax_list( &mdrmax, block_drmax );
        if ( block_need_remake == 1 )
            {
            need_remake = 1;
            }
        }

    }

bool gpu_check_list( tpvec *tcon, tpbox tbox, tplist *tlist )
    {

    need_remake = 0;
    mdrmax = 0.0;

    int block_size = 512;
    dim3 grids( ceil((double)tbox.natom/block_size), 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_check_list <<< grids, threads >>> ( tcon, tbox.natom, tlist );
    cudaDeviceSynchronize();

//    printf( "drmax = %e\n", mdrmax );

    bool flag = 0;
    if ( need_remake )
        flag = 1;

    return flag;
    }
