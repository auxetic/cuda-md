#include "list.h"

// variables define
__managed__ int need_remake;
tpblocks hdblock;

__device__ bool device_nb_ornot( double xi, double yi, double ri, double xj, double yj, double rj, double lx);

// calc parameters of hyperconfig
void calc_nblocks( tpblocks *thdblock, tpbox tbox )
    {
    // numbers of blocks in xy dimension
    int nblockx = ceil( sqrt( (double)tbox.natom / mean_of_block ) );
    int nblocky = nblockx;

    // sum number of blocks
    int nblocks = nblockx * nblocky;

    // length of block
    double dlx = tbox.len.x / nblockx;
    double dly = tbox.len.y / nblockx;

    thdblock->args.nblocks  = nblocks;
    thdblock->args.nblock.x = nblockx;
    thdblock->args.nblock.y = nblocky;
    thdblock->args.dl.x     = dlx;
    thdblock->args.dl.y     = dly;
    }

void recalc_nblocks( tpblocks *thdblock, tpbox tbox )
    {
    // numbers of blocks in xy dimension
    int nblockx = thdblock->args.nblock.x;
    int nblocky = thdblock->args.nblock.y;
    //int nblocky = nblockx;

    // length of block
    double dlx = tbox.len.x / nblockx;
    double dly = tbox.len.y / nblocky;

    thdblock->args.dl.x = dlx;
    thdblock->args.dl.y = dly;
    }

// set block.natom to zero
__global__ void kernel_init_hypercon( tponeblock *tdoneblocks, int tnblocks )
    {
    const int i   = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i < tnblocks )
        tdoneblocks[i].natom = 0;
    }

// kernel subroutine for making hyperconfig
__global__ void kernel_make_hypercon( tponeblock  *tdoneblocks,
                                      int     tnblockx,
                                      double  tdlx,
                                      tpvec   *tdcon,
                                      double  *tdradius,
                                      double  tlx,
                                      int     tnatom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= tnatom )
        return;

    double xi, yi, ri;
    xi = tdcon[i].x;
    yi = tdcon[i].y;
    ri = tdradius[i];

    xi -= round( xi / tlx ) * tlx;
    yi -= round( yi / tlx ) * tlx;

    int bidx, bidy, bid;
    bidx = (int)floor( (xi+0.5*tlx) / tdlx );
    bidy = (int)floor( (yi+0.5*tlx) / tdlx );
    bid  = bidx + bidy * tnblockx;

    int idxtemp = atomicAdd( &tdoneblocks[bid].natom, 1 );

    if ( idxtemp < maxn_of_block-2 )
        {
        tdoneblocks[bid].rx[idxtemp]     = xi;
        tdoneblocks[bid].ry[idxtemp]     = yi;
        tdoneblocks[bid].radius[idxtemp] = ri;
        tdoneblocks[bid].tag[idxtemp]    = i;
        }
    else
        {
        atomicSub( &tdoneblocks[bid].natom, 1 );
        }
    }

// host subroutine used for making hypercon
cudaError_t gpu_make_hypercon( tpblocks thdblock, tpvec *thdcon, double *thdradius, tpbox tbox )
    {
    const int block_size = 256;
    const int nblocks    = thdblock.args.nblocks;
    const int nblockx    = thdblock.args.nblock.x;
    const int natom      = tbox.natom;
    const double lx  = tbox.len.x;
    const double dlx = thdblock.args.dl.x;

    // set hypercon.natom to zero
    dim3 grid1( (nblocks/block_size)+1, 1, 1 );
    dim3 thread1( block_size, 1, 1 );
    kernel_init_hypercon <<< grid1, thread1 >>> ( thdblock.oneblocks, nblocks );
    check_cuda( cudaDeviceSynchronize() );

    // main
    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_make_hypercon <<< grids, threads >>>(thdblock.oneblocks,
                                                nblockx,
                                                dlx,
                                                thdcon,
                                                thdradius,
                                                lx,
                                                natom );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

// set list[].natom to zero; save current config to list.
__global__ void kernel_zero_list( tponelist *tonelist, tpvec *tdcon, int tnatom, double tlx )
    {
    const int i   = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < tnatom )
        {
        tonelist[i].nbsum = 0;
        tonelist[i].x = tdcon[i].x/tlx;
        tonelist[i].y = tdcon[i].y/tlx;
        }
    }

__global__ void kernel_make_list( tponelist *tonelist, tponeblock *toneblocks, double tlx )
    {
    __shared__ tponeblock center_block, nb_block;

    const int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bid  = bidx + bidy * gridDim.x;

    if ( tid == 0 ) 
        center_block = toneblocks[bid];
    __syncthreads();

    for ( int bix=-1; bix<=1; bix++ )
        {
        for ( int biy=-1; biy<=1; biy++ )
            {
            __shared__ intd wrap;
            if ( tid == 0 )
                {
                // get neighbor block
                bidx = blockIdx.x + bix;
                bidy = blockIdx.y + biy;
                // periodic boundary
                wrap.x = (int)floor( (double)bidx / gridDim.x );
                wrap.y = (int)floor( (double)bidy / gridDim.y );
                bidx = bidx - gridDim.x * wrap.x;
                bidy = bidy - gridDim.y * wrap.y;
                bid  = bidx + bidy * gridDim.x;
                nb_block = toneblocks[bid];
                }
            __syncthreads();

            if ( tid < nb_block.natom )
                {
                nb_block.rx[tid] += wrap.x * tlx;
                nb_block.ry[tid] += wrap.y * tlx;
                }
            __syncthreads();

            if ( tid < center_block.natom )
                {
                for ( int jj=0; jj<nb_block.natom; jj++ )
                    {
                    if ( device_nb_ornot(nb_block.rx[jj],nb_block.ry[jj],nb_block.radius[jj],
                                         center_block.rx[tid], center_block.ry[tid], center_block.radius[tid],
                                         tlx) )
                        {
                        int inatom, ilist;
                        inatom = center_block.tag[tid];
                        ilist  = tonelist[inatom].nbsum;
                        if ( ilist == listmax - 2 ) continue;
                        if ( center_block.tag[tid] == nb_block.tag[jj] ) continue;
                        tonelist[inatom].nb[ilist] = nb_block.tag[jj];
                        tonelist[inatom].nbsum += 1;
                        }
                    }
                }
            __syncthreads();
            }
        }
    }

// host subroutine used for makelist
cudaError_t gpu_make_list( tplist   thdlist,
                           tpblocks thdblock,
                           tpvec    *tdcon,
                           tpbox    tbox )
    {
    const int block_size = 256;
    const int natom = tbox.natom;
    const int nblockx = thdblock.args.nblock.x;
    const int nblocky = thdblock.args.nblock.y;
    const double lx = tbox.len.x;

    kernel_zero_list <<< (natom/block_size)+1, block_size >>> ( thdlist.onelists, tdcon, natom, lx );
    check_cuda( cudaDeviceSynchronize() );

    dim3 grids(nblockx,nblocky,1);
    dim3 threads(maxn_of_block,1,1);
    kernel_make_list <<< grids, threads >>> ( thdlist.onelists, thdblock.oneblocks, lx );

    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

__device__ bool device_nb_ornot( double xi, double yi, double ri, 
                                 double xj, double yj, double rj, 
                                 double lx)
    {
    double xij, yij, dij;
    xij = xj - xi;
    yij = yj - yi;
    dij = ri + rj + nlcut;

    xij -= round(xij/lx)*lx;
    yij -= round(yij/lx)*lx;

    double rij2;
    rij2 = xij*xij + yij*yij;

    if ( rij2 < dij*dij )
        return true;
    else
        return false;
    }

__global__ void kernel_make_list_fallback( tponelist *tonelist, tpvec *tdcon, double *tradius, int tnatom, double lx )
    {
    __shared__ double xj[const_256], yj[const_256], rj[const_256];
    const int i   = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;

    int nbsum = 0;

    double xi, yi, ri;
    if ( i < tnatom ) {
        xi = tdcon[i].x; 
        yi = tdcon[i].y; 
        ri = tradius[i];
        }
    else{
        xi = 0.0; 
        yi = 0.0; 
        ri = 0.0; 
        }
    __syncthreads();

    for ( int blocki=0; blocki < ((tnatom/const_256)+1); blocki++ )
        {
        int joffset = blocki * const_256;
        int j = tid + joffset;

        if ( j < tnatom )
            {
            xj[tid] = tdcon[j].x;
            yj[tid] = tdcon[j].y;
            rj[tid] = tradius[j];
            }
        __syncthreads();

        if ( i < tnatom )
            {
            for ( int jj=0; jj<const_256; jj++ )
                {
                if ( jj+joffset >= tnatom ) continue;
                if ( jj+joffset == i ) continue;
                if ( device_nb_ornot(xi,yi,ri,xj[jj],yj[jj],rj[jj],lx) )
                    {
                    if ( nbsum == listmax-2 ) continue;
                    tonelist[i].nb[nbsum] = jj + joffset;
                    nbsum += 1;
                    }
                }
            }
        }
    if ( i < tnatom )
        tonelist[i].nbsum = nbsum;
    }

// host subroutine used for makelist
cudaError_t gpu_make_list_fallback( tplist thdlist, tpvec *thdcon, double *thdradius, tpbox tbox )
    {
    const int block_size = const_256;
    const double lx = tbox.len.x;
    const int natom = tbox.natom;

    kernel_zero_list <<< (natom/block_size)+1, block_size >>> ( thdlist.onelists, thdcon, natom, lx );

    check_cuda( cudaDeviceSynchronize() );

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( maxn_of_block, 1, 1 );
    kernel_make_list_fallback <<< grids, threads >>> ( thdlist.onelists, thdcon, thdradius, natom, lx );

    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

// kernel subroutine used for checkking list
__global__ void kernel_check_list(  tponelist *tonelist,
                                    tpvec     *tdcon,
                                    int       tnatom,
                                    double    lx, 
                                    double    ly )
    {
    __shared__ int block_need_remake;

    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if ( i >= tnatom )
        return;

    if ( tid == 0 )
        block_need_remake = 0;

    __syncthreads();

    double xi, yi;
    xi = tdcon[i].x;
    yi = tdcon[i].y;

    double th_max_dis, dr1, dr2;
    dr1 = fabs(xi/lx - tonelist[i].x)*lx;
    dr2 = fabs(yi/ly - tonelist[i].y)*ly;

    th_max_dis = 2.0 * 1.4 * fmax(dr1, dr2);

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
bool gpu_check_list( tplist thdlist, tpvec *tdcon, tpbox tbox )
    {

    need_remake = 0;

    const int block_size = 512;
    const int natom = tbox.natom;
    const double lx = tbox.len.x;
    const double ly = tbox.len.y;

    dim3  grids( (tbox.natom/block_size)+1, 1, 1 );
    dim3  threads( block_size, 1, 1 );

    kernel_check_list <<< grids, threads >>> ( thdlist.onelists,
                                               tdcon,
                                               natom,
                                               lx, 
                                               ly);
    check_cuda( cudaDeviceSynchronize() );

    bool flag = 0;
    if ( need_remake )
        flag = 1;

    return flag;
    }

int cpu_make_list( tplist tlist, tpvec *tcon, double *tradius, tpbox tbox )
    {
    const int natom = tbox.natom;
    const double lx = tbox.len.x;
    const double ly = tbox.len.y;

    for ( int i=0; i<natom; i++ )
        {
        tlist.onelists[i].nbsum = 0;
        tlist.onelists[i].x = tcon[i].x/lx;
        tlist.onelists[i].y = tcon[i].y/ly;
        }

    for ( int i=0; i<natom; i++ )
        {
        double xi, yi, ri;
        xi = tcon[i].x;
        yi = tcon[i].y;
        ri = tradius[i];
        for ( int j=0; j<natom; j++ )
            {
            if ( i == j ) continue;
            double xj, yj, rj;
            xj = tcon[j].x;
            yj = tcon[j].y;
            rj = tradius[j];

            double xij, yij, dij;
            xij = xj - xi;
            yij = yj - yi;
            xij -= round( xij/lx ) * lx;
            yij -= round( yij/ly ) * ly;
            dij = ri + rj;

            double rij2;
            rij2 = xij*xij + yij*yij;

            double dijprcut2;
            dijprcut2  = dij + nlcut;
            dijprcut2 *= dijprcut2;

            if ( rij2 > dijprcut2 )
                continue;

            int itemp;
            itemp = tlist.onelists[i].nbsum;
            if ( itemp == listmax - 2 ) 
                continue;

            tlist.onelists[i].nb[itemp] = j;
            tlist.onelists[i].nbsum    += 1;
            }
        }
    return 0;
    }
