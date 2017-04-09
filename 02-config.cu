#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct
    {
    int natom;
    double phi;
    double x, y, xinv, yinv;
    double strain;
    int nblocks, nblockx, nblocky;
    double xlim, ylim, bdx, bdy;
    } tpbox;

#define ratio 1.4
#define pi 3.1415926535897932

typedef struct
    {
    int np;
    double phi;
    } tpsets;

typedef struct
    {
    double x, y;
    } tpvec;

#define maxn_of_block 64
typedef struct 
    {
    int bnatom;
    int itag[maxn_of_block];
    double rx[maxn_of_block], ry[maxn_of_block];
    double vx[maxn_of_block], vy[maxn_of_block];
    double fx[maxn_of_block], fy[maxn_of_block];
    double radius[maxn_of_block];
    } tpblock;

void trans_to_hypercon_cpu( tpvec *tcon, double *tradius, tpbox tbox, tpblock *thypercon )
    {
    for ( int i=0; i<tbox.natom; i++ )
        {
        double radius = tradius[i];

        int bix, biy, bid; // store atom block index
        bix = floor( ( tcon[i].x + 0.5 * tbox.x ) / tbox.bdx );
        biy = floor( ( tcon[i].y + 0.5 * tbox.y ) / tbox.bdy );
        bid = bix + biy * tbox.nblockx;
        
        if ( thypercon[bid].bnatom < maxn_of_block )
            {
            int itemp = thypercon[bid].bnatom;
            thypercon[bid].bnatom += 1;
            itemp += 1;
            thypercon[bid].rx[itemp]     = tcon[i].x;
            thypercon[bid].ry[itemp]     = tcon[i].y;
            thypercon[bid].radius[itemp] = tradius[i];
            thypercon[bid].itag[itemp]   = i;
            } else
            {
            printf( "yes, need to think about\n" );
            //
            }
        }
    }

void trans_to_con( tpvec *tcon, double *tradius, tpbox tbox, tpblock *thypercon ) 
    {
    for ( int bid=0; bid<tbox.nblocks; bid++ ) 
        {
        int bnatom = thypercon[bid].bnatom;

        for ( int bi=0; bi<bnatom; bi++ )
            {
            int i = thypercon[bi].itag[bi];
            tcon[i].x = thypercon[bid].rx[bi];
            tcon[i].y = thypercon[bid].ry[bi];
            tradius[i] = thypercon[bid].radius[bi];
            }
        }
    }

void trim_config( tpvec *tcon, tpbox tbox )
    {
    for ( int i=0; i<tbox.natom; i++ )
        {
        tcon[i].x = tcon[i].x - round( tcon[i].x / tbox.x ) * tbox.x;
        tcon[i].y = tcon[i].y - round( tcon[i].y / tbox.y ) * tbox.y;
        }

    }

void gen_config( tpvec *tcon, double *tradius, tpbox tbox, double phi, tpsets sets )
    {
    // 1. intiate random number generator;
    srand(sets.np);

    // 2. set atom radius
    for ( int i=0; i<tbox.natom; i++ )
        {
        if ( i < tbox.natom/2 )
            tradius[i] = 0.5;
        else
            tradius[i] = 0.5 * ratio;
        }

    // 3. cal area of disks
    double sdisk = 0.0;
    for ( int i=0; i<tbox.natom; i++ )
        {
        sdisk += tradius[i]*tradius[i];
        }
    sdisk *= pi;

    // 4. cal box.l from phi
    double vol  = sdisk / phi;
    double temp = sqrt(vol);
    tbox.x    = temp;
    tbox.y    = temp;
    tbox.xinv = 1.0 / temp;
    tbox.yinv = 1.0 / temp;
    tbox.strain = 0.0;

    // 5. give a random config
    for ( int i=0; i<tbox.natom; i++ )
        {
        tcon[i].x = ( (double)rand()/RAND_MAX - 0.5 ) * tbox.x;
        tcon[i].y = ( (double)rand()/RAND_MAX - 0.5 ) * tbox.y;
        }
    }
