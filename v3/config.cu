#include "config.h"

// allocate memory space of config on host
void alloc_con( vec_t **con, double **radius, int natom )
    {
    *con    = (vec_t *)malloc( natom*sizeof(vec_t)  );
    *radius = (double*)malloc( natom*sizeof(double) );
    }

// generate random config on host
void gen_config( vec_t *con, double *radius, box_t *box, sets_t sets )
    {
    // 1. intiate random number generator;
    srand(sets.seed);

    // 2. set atom radius
    for ( int i=0; i<box->natom; i++ )
        {
        if ( i < box->natom/2 )
            radius[i] = 0.5;
        else
            radius[i] = 0.5 * ratio;
        }

    // 3. calc area of disks
    double sdisk = 0.0;
    for ( int i=0; i<box->natom; i++ )
        sdisk += radius[i]*radius[i]*radius[i];
    sdisk *= 4.0 / 3.0 * Pi;

    // 4. cal box.l from phi
    double vol = sdisk / box->phi;
    double lx  = cbrt(vol);
    box->len.x  = lx;
    box->len.y  = lx;
    box->len.z  = lx;
    box->strain = 0.0;

    // 5. give a random config
    for ( int i=0; i<box->natom; i++ )
        {
        con[i].x = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        con[i].y = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        con[i].z = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        }
    }

// calc box length using specific volume fraction
void calc_boxl(double *radius, box_t *box)
    {
    int natom  = box->natom;
    double phi = box->phi;

    double sdisk = 0.0;
    for ( int i=0; i<natom; i++ )
        sdisk += radius[i]*radius[i]*radius[i];
    sdisk *= Pi * 4.0 / 3.0;

    double vol = sdisk / phi;
    double lx  = cbrt(vol);
    box->len.x  = lx;
    box->len.y  = lx;
    box->len.z  = lx;
    box->strain = 0.0;
    }

// generate fcc lattice on host // have some issue // wait for debug
void gen_lattice_fcc ( vec_t *con, double *radius, box_t *box, sets_t sets )
    {
    int natom  = box->natom;
    double phi = box->phi;

    // 1. set atom radius
    for ( int i=0; i<natom; i++ )
        radius[i] = 0.5;

    // 2. calc area of disks
    double sdisk = 0.0;
    for ( int i=0; i<natom; i++ )
        sdisk += radius[i]*radius[i]*radius[i];
    sdisk *= Pi * 4.0 / 3.0;

    // 3. cal box.l from phi
    double vol = sdisk / phi;
    double lx  = cbrt(vol);
    box->len.x  = lx;
    box->len.y  = lx;
    box->len.z  = lx;
    box->strain = 0.0;

    // 4. primitive cell
    int nxyz = (int)cbrt((double)(natom/4));
    if ( 4*(int)pow(nxyz,3) != natom )
        printf("wrong number of atoms\n");
    printf("natom is %d, nxyz is %d\n", natom, nxyz);
    double cl = lx / (double)nxyz;
    printf("primitive cell length is %le\n", cl);
    printf("vol is %le\n", vol);
    printf("phi is %le\n", phi);

    // 5. make lattice
    int i = 0;
    for ( int ii=0; ii<nxyz; ii++ )
        for ( int jj=0; jj<nxyz; jj++ )
            for ( int kk=0; kk<nxyz; kk++ )
                {
                i += 1;
                con[i].x = (ii-1) * cl;
                con[i].y = (jj-1) * cl;
                con[i].z = (kk-1) * cl;
                i += 1;
                con[i].x = (ii-1+0.5) * cl;
                con[i].y = (jj-1+0.5) * cl;
                con[i].z = (kk-1    ) * cl;
                i += 1;
                con[i].x = (ii-1+0.5) * cl;
                con[i].y = (jj-1    ) * cl;
                con[i].z = (kk-1+0.5) * cl;
                i += 1;
                con[i].x = (ii-1    ) * cl;
                con[i].y = (jj-1+0.5) * cl;
                con[i].z = (kk-1+0.5) * cl;
                }

    // 6. add an offset
    double offset = 0.5;
    for ( int i = 0; i < natom; i++ )
        {
        con[i].x += offset;
        con[i].y += offset;
        con[i].z += offset;
        }

    // 7. trim configuration
    //trim_config( con, *box );
    }

// read config
void read_config( FILE *fio, vec_t *con, double *radius, box_t *box )
    {
    double dnatom;
    fscanf(fio, "%le", &dnatom);
    box->natom = (int) dnatom;
    double phi;
    fscanf(fio, "%le", &phi);
    box->phi = phi;
    double tmp;
    fscanf(fio, "%le", &tmp);
    fscanf(fio, "%le", &tmp);
    double lx,ly,lz;
    fscanf(fio, "%le", &lx);
    fscanf(fio, "%le", &ly);
    fscanf(fio, "%le", &lz);
    box->len.x = lx;
    box->len.y = ly;
    box->len.z = lz;
    double strain;
    fscanf(fio, "%le", &strain);
    box->strain = strain;

    for ( int i=0; i<box->natom; i++ )
        {
        double x, y, z, r;
        fscanf(fio, "%le", &x);
        fscanf(fio, "%le", &y);
        fscanf(fio, "%le", &z);
        fscanf(fio, "%le", &r);

        con[i].x  = x;
        con[i].y  = y;
        con[i].z  = z;
        radius[i] = r;
        }
    }

// write config
void write_config( FILE *fio, vec_t *con, double *radius, box_t *box )
    {
    int natom     = box->natom;
    double phi    = box->phi;
    double lx     = box->len.x;
    double ly     = box->len.y;
    double lz     = box->len.z;
    double strain = box->strain;

    //fprintf( fio, "%d %26.16e \n", box.natom, box.len.x );
    fprintf( fio, "%26.16e  %26.16e  %26.16e  %26.16e\n", (double)natom, phi, 0.0, 0.0 );
    fprintf( fio, "%26.16e  %26.16e  %26.16e  %26.16e\n", lx, ly, lz, strain);
    for ( int i = 0; i < natom; i++ )
        fprintf( fio, "%26.16e  %26.16e  %26.16e  %26.16e \n", con[i].x, con[i].y, con[i].z, radius[i] );

    }

// move all atoms to central box
void trim_config( vec_t *con, box_t box )
    {
    double lx = box.len.x;
    double ly = box.len.y;
    double lz = box.len.z;
    for ( int i=0; i<box.natom; i++ )
        {
        double cory;
        cory = round( con[i].y / ly );
        con[i].x -= cory * ly * box.strain;

        con[i].x -= round( con[i].x / lx ) * lx;
        con[i].y -= cory * ly;
        con[i].z -= round( con[i].z / lz ) * lz;
        }
    }


// subroutines for hyperconfiguration
// calc parameters of hyperconfig
void calc_hypercon_args( hycon_t *hycon, box_t box )
    {
    // numbers of blocks in xyz dimension
    int nblockx = ceil( cbrt ( (double)box.natom / mean_size_of_block ) );

    // sum number of blocks
    int nblocks = nblockx * nblockx * nblockx;

    // length of block
    double dlx = box.len.x / nblockx;
    double dly = box.len.y / nblockx;
    double dlz = box.len.z / nblockx;

    if ( dlx < ratio ) printf("#WARNING TOO SMALL HYPERCON BLOCK SIZE FOR FORCE CUTOFF\n");

    hycon->args.nblocks  = nblocks;
    hycon->args.nblock.x = nblocx;
    hycon->args.nblock.y = nblocx;
    hycon->args.nblock.z = nblocx;
    hycon->args.dl.x     = dlx;
    hycon->args.dl.y     = dly;
    hycon->args.dl.z     = dlz;
    }

void recalc_hypercon_args( hycon_t *hycon, box_t box )
    {
    // numbers of blocks in xy dimension
    int nblockx = hycon->args.nblock.x;
    int nblocky = hycon->args.nblock.y;
    int nblockz = hycon->args.nblock.z;

    // length of block
    double dlx = box.len.x / nblockx;
    double dly = box.len.y / nblocky;
    double dlz = box.len.z / nblockz;

    hycon->args.dl.x = dlx;
    hycon->args.dl.y = dly;
    hycon->args.dl.z = dlz;
    }

// calculate minimum image index of block with given x, y, z
__inline__ int indexb( int ix, int iy, int iz, int m)
    {
    return ((ix - 1 + m ) % m )     
           + ((iy - 1 + m ) % m ) * m  
           + ((iz - 1 + m ) % m ) * m * m;
    }

// calculate every block's surrunding 26 block # once and before making hypercon
void map( hycon_t hycon )
    {
    int nblockx   = hycon.args.nblock.x;
    int bid, *neighb;
    for (int ix = 0; ix < nblockx; ix++)
        {
        for (int iy = 0; iy < nblockx; iy++)
            {
            for (int iz = 0; iz < nblockx; iz++)
                {
                    bid        = indexb(ix, iy, iz, nblockx);
                    neighb     = hycon.blocks[bid].neighb;

                    neighb[0]  = indexb(ix + 1, iy,     iz, nblockx); 
                    neighb[1]  = indexb(ix + 1, iy + 1, iz, nblockx);
                    neighb[2]  = indexb(ix    , iy + 1, iz, nblockx);
                    neighb[3]  = indexb(ix - 1, iy + 1, iz, nblockx);
                    neighb[4]  = indexb(ix - 1, iy    , iz, nblockx); 
                    neighb[5]  = indexb(ix - 1, iy - 1, iz, nblockx);
                    neighb[6]  = indexb(ix    , iy - 1, iz, nblockx);
                    neighb[7]  = indexb(ix + 1, iy - 1, iz, nblockx);

                    neighb[8]  = indexb(ix + 1, iy,     iz - 1, nblockx); 
                    neighb[9]  = indexb(ix + 1, iy + 1, iz - 1, nblockx);
                    neighb[10] = indexb(ix    , iy + 1, iz - 1, nblockx);
                    neighb[11] = indexb(ix - 1, iy + 1, iz - 1, nblockx);
                    neighb[12] = indexb(ix - 1, iy    , iz - 1, nblockx); 
                    neighb[13] = indexb(ix - 1, iy - 1, iz - 1, nblockx);
                    neighb[14] = indexb(ix    , iy - 1, iz - 1, nblockx);
                    neighb[15] = indexb(ix + 1, iy - 1, iz - 1, nblockx);
                    neighb[16] = indexb(ix    , iy    , iz - 1, nblockx);

                    neighb[17] = indexb(ix + 1, iy,     iz + 1, nblockx); 
                    neighb[18] = indexb(ix + 1, iy + 1, iz + 1, nblockx);
                    neighb[19] = indexb(ix    , iy + 1, iz + 1, nblockx);
                    neighb[20] = indexb(ix - 1, iy + 1, iz + 1, nblockx);
                    neighb[21] = indexb(ix - 1, iy    , iz + 1, nblockx); 
                    neighb[22] = indexb(ix - 1, iy - 1, iz + 1, nblockx);
                    neighb[23] = indexb(ix    , iy - 1, iz + 1, nblockx);
                    neighb[24] = indexb(ix + 1, iy - 1, iz + 1, nblockx);
                    neighb[25] = indexb(ix    , iy    , iz + 1, nblockx);
                }
            }
        }
    }



// allocate memory space of config on device
cudaError_t alloc_managed_con( vec_t *con, double *radius, int natom )
    {
    check_cuda( cudaMallocManaged( &con,     natom*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &radius , natom*sizeof(double) ) );
    return cudaSuccess;
    }

// allocate memory space of hyperconfig as managed 
cudaError_t alloc_managed_hypercon( hycon_t *hycon )
    {
    int nblocks = hycon->args.nblocks;
    check_cuda( cudaMallocManaged( &hycon->blocks, nblocks*sizeof(block_t) ) );
    return cudaSuccess;
    }

__global__ void kernel_trim_config( vec_t *con, int natom, double lx, double ly, double lz)
    {
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < natom )
        {
        double x, y, z;
        x = con[i].x;
        y = con[i].y;
        z = con[i].z;

        x -= round( x/lx ) * lx;
        y -= round( y/ly ) * ly;
        z -= round( z/lz ) * lz;

        con[i].x = x;
        con[i].y = y;
        con[i].z = z;
        }
    }

cudaError_t gpu_trim_config( vec_t *con, box_t box )
    {
    const int    natom = box.natom;
    const double lx    = box.len.x;
    const double ly    = box.len.y;
    const double lz    = box.len.z;

    const int    block_size = 256;
    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_trim_config<<< grids, threads >>>( con, natom, lx, ly, lz );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

// subroutines for hyperconfiguration

// calculate minimum image index of block with given rx, ry, rz
__inline__ __device__ int indexb( double rx, double ry, double rz, double lx, int m)
    {
    double blocki = (double) m;
    
    rx /= lx;
    ry /= lx;
    rz /= lx;
    rx -= round(rx);
    ry -= round(ry);
    rz -= round(rz);

    return  (int)floor((rx + 0.5) * blocki )     
           +(int)floor((ry + 0.5) * blocki ) * m
           +(int)floor((rz + 0.5) * blocki ) * m * m;
    }


__global__ void kernel_reset_hypercon_block(block_t *block)
    {
    const int i = threadIdx.x;

    block->rx[i]     = 0.0;
    block->ry[i]     = 0.0;
    block->rz[i]     = 0.0;
    block->radius[i] = 0.0;
    block->tag[i]    = -1;
    }

// calculte index of each atom and register it into block structure // map config into hyperconfig
__global__ void kernel_make_hypercon( block_t *blocks,
                                      vec_t   *con,
                                      double  *radius,
                                      double  lx,
                                      int     nblockx,
                                      int     natom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= natom ) return;

    double rx = con[i].x;
    double ry = con[i].y;
    double rz = con[i].z;
    double ri = radius[i];

    double x = rx;
    double y = ry;
    double z = rz;

    rx /= lx;
    ry /= lx;
    rz /= lx;
    rx -= round(rx);
    ry -= round(ry);
    rz -= round(rz);

    int bid;
    bid =  (int) floor((rx + 0.5) * (double)nblockx )                          
          +(int) floor((ry + 0.5) * (double)nblockx ) * nblockx             
          +(int) floor((rz + 0.5) * (double)nblockx ) * nblockx * nblockx;

    int idxinblock = atomicAdd( &blocks[bid].natom, 1);
    if ( idxinblock < max_size_of_block - 2 )
        {
        blocks[bid].rx[idxinblock]     = x;
        blocks[bid].ry[idxinblock]     = y;
        blocks[bid].rz[idxinblock]     = z;
        blocks[bid].radius[idxinblock] = ri;
        blocks[bid].tag[idxinblock]    = i;
        }
    else
        {
        // TODO one should consider those exced the max number of atoms per blocks
        atomicSub( &blocks[bid].natom, 1 );
        //idxinblock = atomicAdd( &blocks[bid]->extra.natom, 1);
        //blocks[bid]->extra.rx[idxinblock]     = x;
        //blocks[bid]->extra.ry[idxinblock]     = y;
        //blocks[bid]->extra.rz[idxinblock]     = z;
        //blocks[bid]->extra.radius[idxinblock] = ri;
        //blocks[bid]->extra.tag[idxinblock]    = i;
        }
    }

cudaError_t gpu_make_hypercon( hycon_t hycon, vec_t *con, double *radius, box_t box)
    {
    const int block_size = 256;
    const int nblocks    = hycon.args.nblocks;
    const int nblockx    = hycon.args.nblock.x;
    const int natom      = box.natom;
    const double lx      = box.len.x;

    int grids, threads;

    //reset hypercon
    block_t *block;
    for ( int i = 0; i < nblocks; i++ )
        {
        block = &hycon.blocks[i];

        block->natom = 0;
        threads = max_size_of_block;
        kernel_reset_hypercon_block <<<1, threads>>> (block);
        }
    check_cuda( cudaDeviceSynchronize() );

    // recalculate hypercon block length
    //recalc_hypercon_args(hycon, box );

    // main
    grids   = (natom/block_size)+1;
    threads = block_size;

    kernel_make_hypercon <<< grids, threads >>>(hycon.blocks,
                                                con,
                                                radius,
                                                lx,
                                                nblockx,
                                                natom );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_map_hypercon_con (block_t *block, vec_t *con, double *radius)
    {
    const int tid = threadIdx.x;

    const int i = block->tag[tid];

    if ( tid >= block->natom) return;

    con[i].x  = block->rx[tid];
    con[i].y  = block->ry[tid];
    con[i].z  = block->rz[tid];
    radius[i] = block->radius[tid];
    }

cudaError_t gpu_map_hypercon_con( hycon_t *hycon, vec_t *con, double *radius, box_t box)
    {
    const int nblocks    = hycon->args.nblocks;
    // const int nblockx = hycon->args.nblock.x;
    // const int natom   = box.natom;
    // const double lx   = box.len.x;

    //map hypercon into normal con with index of atom unchanged
    int grids, threads;
    block_t *block;
    for ( int i = 0; i < nblocks; i++ )
        {
        block = &hycon->blocks[i];
        grids = 1;
        threads = max_size_of_block;
        kernel_map_hypercon_con<<<grids, threads>>>(block, con, radius);
        }
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }
