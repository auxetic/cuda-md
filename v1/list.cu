#include "list.h"

// variables define
__managed__ int need_remake;
tpblockset hblockset;
tpblock *dblocks;
tplist  *dlist;

// calc parameters of hyperconfig
void calc_nblocks( tpblockset *tblockset, tpbox tbox )
    {
    // numbers of blocks in xy dimension
    int nblockx = ceil( sqrt( (double)tbox.natom / mean_of_block ) );
    int nblocky = nblockx;

    // sum number of blocks
    int nblocks = nblockx * nblocky;

    // length of block
    double dlx = tbox.x / nblockx;
    double dly = tbox.y / nblockx;

    tblockset->nblocks = nblocks;
    tblockset->nblockx = nblockx;
    tblockset->nblocky = nblocky;
    tblockset->dlx     = dlx;
    tblockset->dly     = dly;
    }

void recalc_nblocks( tpblockset *tblockset, tpbox tbox )
    {
    // numbers of blocks in xy dimension
    int nblockx = tblockset->nblockx;
    int nblocky = tblockset->nblocky;
    //int nblocky = nblockx;

    // length of block
    double dlx = tbox.x / nblockx;
    double dly = tbox.y / nblocky;

    tblockset->dlx     = dlx;
    tblockset->dly     = dly;
    }

// set block.natom to zero
__global__ void kernel_init_hypercon( tpblock *tdblocks, int tnblocks )
    {
    const int i   = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i < tnblocks )
        tdblocks[i].natom = 0;
    }

// kernel subroutine for making hyperconfig
__global__ void kernel_make_hypercon( tpblock *tdblocks,
                                      int     tnblockx,
                                      double  dlx,
                                      double  dly,
                                      tpvec   *tdcon,
                                      double  *tdradius,
                                      double  lx,
                                      double  ly,
                                      int     tnatom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= tnatom )
        return;

    float xi, yi, ri;
    xi = tdcon[i].x;
    yi = tdcon[i].y;
    ri = tdradius[i];

    xi -= round( xi / lx ) * lx;
    yi -= round( yi / ly ) * ly;

    int bidx, bidy, bid;
    bidx = floor( (xi+0.5*lx) / dlx );
    bidy = floor( (yi+0.5*ly) / dly );
    bid  = bidx + bidy * tnblockx;

    int idxtemp = atomicAdd( &tdblocks[bid].natom, 1 );

    if ( idxtemp < maxn_of_block-2 )
        {
        tdblocks[bid].rx[idxtemp]     = xi;
        tdblocks[bid].ry[idxtemp]     = yi;
        tdblocks[bid].radius[idxtemp] = ri;
        tdblocks[bid].tag[idxtemp]    = i;
        }
    else
        {
        atomicSub( &tdblocks[bid].natom, 1 );
        }
    }

// host subroutine used for making hypercon
cudaError_t gpu_make_hypercon( tpvec *tdcon, double *tdradius, tpbox tbox, tpblock *tdblocks, tpblockset tblockset )
    {
    const int  block_size = 256;

    // set hypercon.natom to zero
    dim3 grid1( (tblockset.nblocks/block_size)+1, 1, 1 );
    dim3 thread1( block_size, 1, 1 );
    kernel_init_hypercon <<< grid1, thread1 >>> ( tdblocks, tblockset.nblocks );

    cudaError_t err;
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    // main
    dim3 grids( (tbox.natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_make_hypercon <<< grids, threads >>>( tdblocks,
                                                 tblockset.nblockx,
                                                 tblockset.dlx,
                                                 tblockset.dly,
                                                 tdcon,
                                                 tdradius,
                                                 tbox.x,
                                                 tbox.y,
                                                 tbox.natom );

    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }

// set list[].natom to zero; save current config to list.
__global__ void kernel_zero_list( tpvec *tdcon, tplist *tdlist, int natom, double lx, double ly )
    {
    const int i   = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < natom )
        {
        tdlist[i].nbsum = 0;
        tdlist[i].x = tdcon[i].x/lx;
        tdlist[i].y = tdcon[i].y/ly;
        }
    }

__global__ void kernel_make_list( tplist *tdlist,
                                  float lx,
                                  float ly,
                                  tpblock *tdblocks )
    {
    __shared__ tpblock local_blocks[9];

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

        local_blocks[atomi] = tdblocks[joffset];
        }

    __syncthreads();

    if ( atomi < local_blocks[4].natom )
        {
        float xi, yi, ri;
        int itag;
        // get xyr from shared memory
        xi   = local_blocks[4].rx[atomi];
        yi   = local_blocks[4].ry[atomi];
        ri   = local_blocks[4].radius[atomi];
        itag = local_blocks[4].tag[atomi];
        for ( int ib=0; ib<9; ib++ )
            {
            for ( int atomj=0; atomj<local_blocks[ib].natom; atomj++ )
                {
                float xj, yj, rj;
                int jtag;
                xj   = local_blocks[ib].rx[atomj];
                yj   = local_blocks[ib].ry[atomj];
                rj   = local_blocks[ib].radius[atomj];
                jtag = local_blocks[ib].tag[atomj];
                if ( itag == jtag ) continue;

                float xij, yij, rij2;
                xij  = xj - xi;
                yij  = yj - yi;
                xij -= round(xij/lx) * lx;
                yij -= round(yij/ly) * ly;
                rij2 = xij*xij + yij*yij;

                float dijcut2;
                dijcut2  = ri + rj + nlcut;
                dijcut2 *= dijcut2;

                if ( rij2 < dijcut2 )
                    {
                    if ( tdlist[itag].nbsum == listmax - 2 ) continue;
                    tdlist[itag].nbsum += 1;
                    tdlist[itag].nb[tdlist[itag].nbsum-1] = jtag;
                    }
                }
            }
        }
    }

// host subroutine used for makelist
cudaError_t gpu_make_list( tplist  *tdlist,
                           tpblock *tdblocks,
                           tpvec   *tdcon,
                           tpblockset tblockset,
                           tpbox tbox )
    {
    const int block_size = 256;
    float lx = tbox.x;
    float ly = tbox.y;

    kernel_zero_list <<< (tbox.natom/block_size)+1, block_size >>> ( tdcon, tdlist, tbox.natom, tbox.x, tbox.y );
    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    dim3 grids(tblockset.nblockx,tblockset.nblocky,1);
    dim3 threads(maxn_of_block,1,1);
    kernel_make_list <<< grids, threads >>> ( tdlist, lx, ly, tdblocks );

    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }


// kernel subroutine used for checkking list
__global__ void kernel_check_list(  tpvec *tdcon,
                                    int tnatom,
                                    tplist *tdlist,
                                    float lx, float ly )
    {
    __shared__ int    block_need_remake;

    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if ( i >= tnatom )
        return;

    if ( tid == 0 )
        {
        block_need_remake = 0;
        }

    __syncthreads();

    float xi, yi;
    xi = tdcon[i].x;
    yi = tdcon[i].y;

    double th_max_dis, dr1, dr2; //dr3;
    dr1 = (xi/lx - tdlist[i].x)*lx;
    dr2 = (yi/ly - tdlist[i].y)*ly;

    th_max_dis = 2.0 * 1.42 * fmax(dr1, dr2);

    if ( th_max_dis > nlcut )
        block_need_remake = 1;

    __syncthreads();

    if ( tid == 0 )
        {
        if ( block_need_remake == 1 )
            need_remake = 1;
        }

    }

// host subroutine used for checking list
bool gpu_check_list( tpvec *tdcon, tpbox tbox, tplist *tdlist )
    {

    need_remake = 0;

    float lx = tbox.x;
    float ly = tbox.y;

    int block_size = 512;
    dim3 grids( (tbox.natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_check_list <<< grids, threads >>> ( tdcon,
                                               tbox.natom,
                                               tdlist,
                                               lx, ly );
    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    bool flag = 0;
    if ( need_remake )
        flag = 1;

    return flag;
    }
