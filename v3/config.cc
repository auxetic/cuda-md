#include "config.h"

// allocate memory space of config on host
void alloc_con( vec_t **tcon, double **tradius, int tnatom )
    {
    *tcon    = (vec_t *)malloc( tnatom*sizeof(vec_t)  );
    *tradius = (double*)malloc( tnatom*sizeof(double) );
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
        sdisk += tradius[i]*tradius[i]*tradius[i];
        }
    sdisk *= 4.0 / 3.0 * Pi;


    // 4. cal box.l from phi
    double vol     = sdisk / tbox->phi;
    double lx      = cbrt(vol);
    tbox->len.x    = lx;
    tbox->len.y    = lx;
    tbox->len.z    = lx;
    tbox->leninv.x = 1.0 / lx;
    tbox->leninv.y = 1.0 / lx;
    tbox->leninv.z = 1.0 / lx;
    tbox->strain   = 0.0;

    // 5. give a random config
    for ( int i=0; i<tbox->natom; i++ )
        {
        tcon[i].x = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        tcon[i].y = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        tcon[i].z = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
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
// generate fcc lattice on host // have some issue // wait for debug
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
    double lz = tbox.len.z;
    for ( int i=0; i<tbox.natom; i++ )
        {
        double cory;
        cory = round( tcon[i].y / ly );
        tcon[i].x -= cory * ly * tbox.strain;

        tcon[i].x -= round( tcon[i].x / lx ) * lx;
        tcon[i].y -= cory * ly;
        tcon[i].z -= round( tcon[i].z / lz ) * lz;
        }
    }


// subroutines for hyperconfiguration
// calc parameters of hyperconfig
void calc_hypercon_args( hycon_t *thdhycon, box_t tbox )
    {
    // numbers of blocks in xyz dimension
    int nblockx = ceil( cbrt ( (double)tbox.natom / mean_size_of_block ) );

    // sum number of blocks
    int nblocks = nblockx * nblockx * nblockx;

    // length of block
    double dlx = tbox.len.x / nblockx;
    double dly = tbox.len.y / nblockx;
    double dlz = tbox.len.z / nblockx;

    if ( dlx < ratio ) printf("#WARNING TOO SMALL HYPERCON BLOCK SIZE FOR FORCE CUTOFF\n");

    thdhycon->args.nblocks  = nblocks;
    thdhycon->args.nblock.x = nblockx;
    thdhycon->args.nblock.y = nblockx;
    thdhycon->args.nblock.z = nblockx;
    thdhycon->args.dl.x     = dlx;
    thdhycon->args.dl.y     = dly;
    thdhycon->args.dl.z     = dlz;
    }

void recalc_hypercon_args( hycon_t *thdblock, box_t tbox )
    {
    // numbers of blocks in xy dimension
    int nblockx = thdblock->args.nblock.x;
    int nblocky = thdblock->args.nblock.y;
    int nblockz = thdblock->args.nblock.z;

    // length of block
    double dlx = tbox.len.x / nblockx;
    double dly = tbox.len.y / nblocky;
    double dlz = tbox.len.z / nblockz;

    thdblock->args.dl.x = dlx;
    thdblock->args.dl.y = dly;
    thdblock->args.dl.z = dlz;
    }

// calculate minimum image index of block with given x, y, z
__inline__ int indexb( int ix, int iy, int iz, int m)
    {
    return ((ix - 1 + m ) % m )     
           ((iy - 1 + m ) % m ) * m  
           ((iz - 1 + m ) % m ) * m * m;
    }

// calculate every block's surrunding 26 block # once and before making hypercon
void map( hycon_t thdblocks )
    {
    int nblockx   = thdblocks.args.nblock.x;
    int bid, *neighb;
    for (int ix = 0; ix < nblockx; ix++)
        {
        for (int iy = 0; iy < nblockx; iy++)
            {
            for (int iz = 0; iz < nblockx; iz++)
                {
                    bid        = indexb(ix, iy, iz, nblockx);
                    neighb     = thdblocks.oneblocks[bid].neighb;

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


