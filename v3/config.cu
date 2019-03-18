#include "config.h"

// variables define
// host
vec_t  *con;
double *radius;
// device
vec_t  *dcon;
double *dradius;


// allocate memory space of config on host
void alloc_con( vec_t **tcon, double **tradius, int tnatom )
    {
    *tcon    = (vec_t *)malloc( tnatom*sizeof(vec_t)  );
    *tradius = (double*)malloc( tnatom*sizeof(double) );
    }

// allocate memory space of config on device
cudaError_t device_alloc_con( vec_t **tcon, double **tradius, int tnatom )
    {
    check_cuda( cudaMallocManaged( &tcon,     tnatom*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &tradius , tnatom*sizeof(double) ) );
    return cudaSuccess;
    }

// allocate memory space of hyperconfig as managed 
cudaError_t alloc_hypercon( hycon_t *thdhycon )
    {
    int nblocks = thdhycon->args.nblocks;
    check_cuda( cudaMallocManaged( &thdhycon.oneblocks, nblocks*sizeof(cell_t) ) );
    return cudaSuccess;
    }

// generate random config on host
void gen_config( vec_t *tcon, double *tradius, box_t *tbox, sets_t tsets )
    {
    // 1. intiate random number generator;
    srand(tsets.seed);

    // 2. set atom radius
    for ( int i=0; i<tbox->natom; i++ )
        {
        if ( i < tbox->natom/2 )
            tradius[i] = 0.5;
        else
            tradius[i] = 0.5 * ratio;
        }

    // 3. calc area of disks
    double sdisk = 0.0;
    for ( int i=0; i<tbox->natom; i++ )
        {
        sdisk += tradius[i]*tradius[i];
        }
    sdisk *= Pi;

#if sysdim == 3
    sdisk = 0.0;
    for ( int i=0; i<tbox->natom; i++ )
        {
        sdisk += tradius[i]*tradius[i]*tradius[i];
        }
    sdisk *= 4.0 / 3.0 * Pi;
#endif


    // 4. cal box.l from phi
    double vol     = sdisk / tbox->phi;
    double lx      = sqrt(vol);
    tbox->len.x    = lx;
    tbox->len.y    = lx;
    tbox->leninv.x = 1.0 / lx;
    tbox->leninv.y = 1.0 / lx;
    tbox->strain   = 0.0;

#if sysdim == 3
    lx             = cbrt(vol);
    tbox->len.x    = lx;
    tbox->len.y    = lx;
    tbox->len.z    = lx;
    tbox->leninv.x = 1.0 / lx;
    tbox->leninv.y = 1.0 / lx;
    tbox->leninv.z = 1.0 / lx;
    tbox->strain   = 0.0;
#endif

    // 5. give a random config
    for ( int i=0; i<tbox->natom; i++ )
        {
        tcon[i].x = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        tcon[i].y = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
#if sysdim == 3
        tcon[i].z = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
#endif
        }
    }

// calc box length using specific volume fraction
void calc_boxl(double *tradius, box_t *tbox)
    {
    int natom  = tbox->natom;
    double phi = tbox->phi;

    double sdisk = 0.0;
    for ( int i=0; i<natom; i++ )
        sdisk += tradius[i]*tradius[i]*tradius[i];
    sdisk *= Pi * 4.0 / 3.0;

    double vol     = sdisk / phi;
    double lx      = cbrt(vol);
    tbox->len.x    = lx;
    tbox->len.y    = lx;
    tbox->len.z    = lx;
    tbox->leninv.x = 1.0 / lx;
    tbox->leninv.y = 1.0 / lx;
    tbox->leninv.z = 1.0 / lx;
    tbox->strain   = 0.0;
    }
// generate fcc lattice on host
// have some issue
// wait for debug
void gen_lattice_fcc ( vec_t *tcon, double *tradius, box_t *tbox, sets_t tsets )
    {
    int natom  = tbox->natom;
    double phi = tbox->phi;

    // 1. set atom radius
    for ( int i=0; i<natom; i++ )
            tradius[i] = 0.5;

    // 2. calc area of disks
    double sdisk = 0.0;
    for ( int i=0; i<natom; i++ )
        sdisk += tradius[i]*tradius[i]*tradius[i];
    sdisk *= Pi * 4.0 / 3.0;

    // 3. cal box.l from phi
    double vol     = sdisk / phi;
    double lx      = cbrt(vol);
    tbox->len.x    = lx;
    tbox->len.y    = lx;
    tbox->len.z    = lx;
    tbox->leninv.x = 1.0 / lx;
    tbox->leninv.y = 1.0 / lx;
    tbox->leninv.z = 1.0 / lx;
    tbox->strain   = 0.0;

    // 4. primitive cell
    int nxyz = (int)cbrt((double)(natom/4));
    if ( 4*(int)pow(nxyz,3) != natom )
        printf("wrong number of atoms\n");
    printf("natom is %d, nxyz is %d\n", natom, nxyz);
    double cl;cl = lx / (double)nxyz;
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
                tcon[i].x = (ii-1) * cl;
                tcon[i].y = (jj-1) * cl;
                tcon[i].z = (kk-1) * cl;
                i += 1;
                tcon[i].x = (ii-1+0.5) * cl;
                tcon[i].y = (jj-1+0.5) * cl;
                tcon[i].z = (kk-1    ) * cl;
                i += 1;
                tcon[i].x = (ii-1+0.5) * cl;
                tcon[i].y = (jj-1    ) * cl;
                tcon[i].z = (kk-1+0.5) * cl;
                i += 1;
                tcon[i].x = (ii-1    ) * cl;
                tcon[i].y = (jj-1+0.5) * cl;
                tcon[i].z = (kk-1+0.5) * cl;
                }

    // 6. add an offset
    double offset = 0.5;
    for ( int i = 0; i < natom; i++ )
        {
        tcon[i].x += offset;
        tcon[i].y += offset;
        tcon[i].z += offset;
        }

    // 7. trim configuration
    //trim_config( tcon, *tbox );
    }

// read config
void read_config( FILE *tfio, vec_t *tcon, double *tradius, box_t *tbox )
    {
    double dnatom;
    fscanf(tfio, "%le", &dnatom);
    tbox->natom = (int) dnatom;
    double phi;
    fscanf(tfio, "%le", &phi);
    tbox->phi = phi;
    double tmp;
    fscanf(tfio, "%le", &tmp);
    fscanf(tfio, "%le", &tmp);
    double lx,ly,lz;
    fscanf(tfio, "%le", &lx);
    fscanf(tfio, "%le", &ly);
    fscanf(tfio, "%le", &lz);
    tbox->len.x = lx;
    tbox->len.y = ly;
    tbox->len.z = lz;
    tbox->leninv.x = 1e0/lx;
    tbox->leninv.y = 1e0/ly;
    tbox->leninv.z = 1e0/lz;
    double strain;
    fscanf(tfio, "%le", &strain);
    tbox->strain = strain;

    for ( int i=0; i<tbox->natom; i++ )
        {
        double x, y, z, r;
        fscanf(tfio, "%le", &x);
        fscanf(tfio, "%le", &y);
        fscanf(tfio, "%le", &z);
        fscanf(tfio, "%le", &r);

        tcon[i].x  = x;
        tcon[i].y  = y;
        tcon[i].z  = z;
        tradius[i] = r;
        }
    }

// write config
void write_config( FILE *tfio, vec_t *tcon, double *tradius, box_t *tbox )
    {
    int natom      = tbox->natom;
    double phi     = tbox->phi;
    double lx      = tbox->len.x;
    double ly      = tbox->len.y;
    double lz      = tbox->len.z;
    double strain  = tbox->strain;

        //fprintf( fio, "%d %26.16e \n", box.natom, box.len.x );
    fprintf( tfio, "%26.16e  %26.16e  %26.16e  %26.16e\n", (double)natom, phi, 0.0, 0.0 );
    fprintf( tfio, "%26.16e  %26.16e  %26.16e  %26.16e\n", lx, ly, lz, strain);
    for ( int i = 0; i < natom; i++ )
        fprintf( tfio, "%26.16e  %26.16e  %26.16e  %26.16e \n", tcon[i].x, tcon[i].y, tcon[i].z, tradius[i] );

    }

// move all atoms to central box
void trim_config( vec_t *tcon, box_t tbox )
    {
    double lx = tbox.len.x;
    double ly = tbox.len.y;
#if sysdim == 3
    double lz = tbox.len.z;
#endif
    for ( int i=0; i<tbox.natom; i++ )
        {
        double cory;
        cory = round( tcon[i].y / ly );
        tcon[i].x -= cory * ly * tbox.strain;

        tcon[i].x -= round( tcon[i].x / lx ) * lx;
        tcon[i].y -= cory * ly;
#if sysdim == 3
        tcon[i].z -= round( tcon[i].z / lz ) * lz;
#endif
        }
    }

__global__ void kernel_trim_config( vec_t *tcon, int tnatom, double lx, double ly, double lz)
    {
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < tnatom )
        {
        double x, y, z;
        x = tcon[i].x;
        y = tcon[i].y;

        x -= round( x/lx ) * lx;
        y -= round( y/ly ) * ly;

        tcon[i].x = x;
        tcon[i].y = y;

#if sysdim == 3
        z = tcon[i].z;
        z -= round( z/lz ) * lz;
        tcon[i].z = z;
#endif
        }
    }

cudaError_t gpu_trim_config( vec_t *tcon, box_t tbox )
    {
    const int    natom = tbox.natom;
    const double lx    = tbox.len.x;
    const double ly    = tbox.len.y;
    const double lz    = tbox.len.z;

    const int    block_size = 256;
    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_trim_config<<< grids, threads >>>( tcon, natom, lx, ly, lz );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

// calculate minimum image index of block with given x, y, z
__inline__ int indexb( int ix, int iy, int iz, int m)
    {
    return ((ix - 1 + m ) % m )     \
           ((iy - 1 + m ) % m ) * m \
           ((iz - 1 + m ) % m ) * m * m;
    }

// calculate every block's surrunding 26 block # once and before making hypercon
void map( hycon_t thdblocks )
    {
    int nblockx   = thdblocks.args.nblockx;
    int bid, *neighb;
    for (int ix = 0; ix < nblockx; ix++)
        {
        for (int iy = 0; iy < nblockx; iy++)
            {
            for (int iz = 0; iz < nblockx; iz++)
                {
                    bid        = indexb(ix, iy, iz, nblockx);
                    neighb     = thdblocks.oneblocks[bid].neighb;

                    neighb[0]  = indexb(ix + 1, iy,     iz   ); 
                    neighb[1]  = indexb(ix + 1, iy + 1, iz   );
                    neighb[2]  = indexb(ix    , iy + 1, iz   );
                    neighb[3]  = indexb(ix - 1, iy + 1, iz   );
                    neighb[4]  = indexb(ix - 1, iy    , iz   ); 
                    neighb[5]  = indexb(ix - 1, iy - 1, iz   );
                    neighb[6]  = indexb(ix    , iy - 1, iz   );
                    neighb[7]  = indexb(ix + 1, iy - 1, iz   );

                    neighb[8]  = indexb(ix + 1, iy,     iz - 1); 
                    neighb[9]  = indexb(ix + 1, iy + 1, iz - 1);
                    neighb[10] = indexb(ix    , iy + 1, iz - 1);
                    neighb[11] = indexb(ix - 1, iy + 1, iz - 1);
                    neighb[12] = indexb(ix - 1, iy    , iz - 1); 
                    neighb[13] = indexb(ix - 1, iy - 1, iz - 1);
                    neighb[14] = indexb(ix    , iy - 1, iz - 1);
                    neighb[15] = indexb(ix + 1, iy - 1, iz - 1);
                    neighb[16] = indexb(ix    , iy    , iz - 1);

                    neighb[17] = indexb(ix + 1, iy,     iz + 1); 
                    neighb[18] = indexb(ix + 1, iy + 1, iz + 1);
                    neighb[19] = indexb(ix    , iy + 1, iz + 1);
                    neighb[20] = indexb(ix - 1, iy + 1, iz + 1);
                    neighb[21] = indexb(ix - 1, iy    , iz + 1); 
                    neighb[22] = indexb(ix - 1, iy - 1, iz + 1);
                    neighb[23] = indexb(ix    , iy - 1, iz + 1);
                    neighb[24] = indexb(ix + 1, iy - 1, iz + 1);
                    neighb[25] = indexb(ix    , iy    , iz + 1);
                }
            }
        }
    }

// calculate minimum image index of block with given rx, ry, rz
__inline__ __device__ int indexb( double rx, double ry, double rz, double tlx, int m)
    {
    double blocki = (double) m
    
    rx /= tlx;
    ry /= tlx;
    rz /= tlx;
    rx -= round(rx);
    ry -= round(ry);
    rz -= round(rz);

    return (int)floor((rx + 0.5) * blocki )     \
           (int)floor((ry + 0.5) * blocki ) * m \
           (int)floor((rz + 0.5) * blocki ) * m * m;
    }


__global__ void kernel_reset_hypercon_block(cell_t *block)
    {
    const int i = threadIdx.x;

    block->rx[i]     = 0.0;
    block->ry[i]     = 0.0;
    block->rz[i]     = 0.0;
    block->radius[i] = 0.0;
    block->tag[i]    = -1;
    }

// calculte index of each atom and register it into block structure // map config into hyperconfig
__global__ void kernel_make_hypercon( cell_t *tdoneblocks,
                                      vec_t      *tdcon,
                                      double     *tdradius,
                                      double      tlx,
                                      int         tnblockx,
                                      int         tnatom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= tnatom ) return;

    double rx = tdcon[i].x;
    double ry = tdcon[i].y;
    double rz = tdcon[i].z;
    double ri = tdradius[i];

    double x = rx;
    double y = ry;
    double z = rz;

    rx /= tlx;
    ry /= tlx;
    rz /= tlx;
    rx -= round(rx);
    ry -= round(ry);
    rz -= round(rz);

    int bid;
    bid = (int) floor((rx + 0.5) * (double)tnblockx )                          
          (int) floor((ry + 0.5) * (double)tnblockx ) * tnblockx             
          (int) floor((rz + 0.5) * (double)tnblockx ) * tnblockx * tnblockx;

    int idxinblock = atomicAdd( &tdoneblocks[bid].natom, 1);
    if ( idxinblock < max_size_of_block - 2 )
        {
        tdoneblocks[bid].rx[idxinblock]     = x;
        tdoneblocks[bid].ry[idxinblock]     = y;
        tdoneblocks[bid].rz[idxinblock]     = z;
        tdoneblocks[bid].radius[idxinblock] = ri;
        tdoneblocks[bid].tag[idxinblock]    = i;
        }
    else
        {
        // TODO one should consider those exced the max number of atoms per blocks
        atomicSub( &tdoneblocks[bid].natom, 1 );
        idxinblock = atomicAdd( &tdoneblocks[bid]->extrablock.natom, 1);
        tdoneblocks[bid]->extrablock.rx[idxinblock]     = x;
        tdoneblocks[bid]->extrablock.ry[idxinblock]     = y;
        tdoneblocks[bid]->extrablock.rz[idxinblock]     = z;
        tdoneblocks[bid]->extrablock.radius[idxinblock] = ri;
        tdoneblocks[bid]->extrablock.tag[idxinblock]    = i;
        }
    }

cudaError_t gpu_make_hypercon( hycon_t *thdblock, vec_t *thdcon, double *thdradius, box_t tbox)
    {
    const int block_size = 256;
    const int nblocks    = thdblock->args.nblocks;
    const int nblockx    = thdblock->args.nblock.x;
    const int natom      = tbox.natom;
    const double lx      = tbox.len.x;

    int grids, threads;

    //reset hypercon
    cell_t *block;
    for ( int i = 0; i < nblocks; i++ )
        {
        block = &thdblock->oneblocks[i];

        block->natom = 0;
        threads = max_size_of_block;
        kernel_reset_hypercon_block <<<1, threads>>> (block);
        }
    check_cuda( cudaDeviceSynchronize() );

    // recalculate hypercon block length
    recalc_nblocks( thdblock, tbox );

    // main
    grids   = (natom/block_size)+1;
    threads = block_size;
    kernel_make_hypercon <<< grids, threads >>>(thdblock->oneblocks,
                                                nblockx,
                                                thdcon,
                                                thdradius,
                                                lx,
                                                natom );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }


__global__ void kernel_map_hypercon_con (cell_t *tblock, vec_t *tdcon, vec_t *tdradius)
    {
    const int tid = threadIdx.x;

    const int i   = tblock->tag[tid];

    if ( tid >= tblock->natom) return;

    tdcon[i].x = tblock->rx[tid];
    tdcon[i].y = tblock->ry[tid];
    tdcon[i].z = tblock->rz[tid];
    tdradius[i].x = tblock->radius[tid];

    }
cudaError_t gpu_map_hypercon_con( hycon_t *thdblock, vec_t *thdcon, double *thdradius, box_t tbox)
    {
    const int nblocks    = thdblock->args.nblocks;
    const int nblockx    = thdblock->args.nblock.x;
    const int natom      = tbox.natom;
    const double lx      = tbox.len.x;

    //map hypercon into normal con with index of atom unchanged
    int grids, threads;
    for ( int i = 0; i < nblocks; i++ )
        {
        block = &thdblock->oneblocks[i];
        threads = max_size_of_block;
        kernel_map_hypercon_con<<<1, threads>>>(block, thdcon, thdradius);
        }
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }
