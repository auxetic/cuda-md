#include "config.h"

// variables define
// host
tpvec *con; 
double *radius;
// device
tpvec *dcon;
double *dradius;


void alloc_con( tpvec **tcon, double **tradius, int natom )
    {
    *tcon    = (tpvec *)malloc(sizeof(tpvec)*natom);
    *tradius = (double*)malloc(sizeof(double)*natom);
    }

cudaError_t device_alloc_con( tpvec **tcon, double **tradius, int natom )
    {
    cudaError_t err;
    err = cudaMalloc( (void **)tcon    , natom*sizeof(tpvec) );
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"Malloc failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
        }
    err = cudaMalloc( (void **)tradius , natom*sizeof(double) );
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"Malloc failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
        }
    return cudaSuccess;
    }

cudaError_t trans_con_to_gpu( tpvec *thcon, double *thradius, int natom,
                              tpvec *tdcon, double *tdradius )
    {
    cudaError_t err;
    err = cudaMemcpy(tdcon, thcon, natom*sizeof(tpvec), cudaMemcpyHostToDevice);
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"cudaMemcpy failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
        }
    err = cudaMemcpy(tdradius, thradius, natom*sizeof(double), cudaMemcpyHostToDevice);
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"cudaMemcpy failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
        }
    return cudaSuccess;
    }

cudaError_t trans_con_to_host( tpvec *tdcon, double *tdradius, int natom,
                               tpvec *thcon, double *thradius )
    {
    cudaError_t err;
    err = cudaMemcpy(thcon, tdcon, natom*sizeof(tpvec), cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"cudaMemcpy failed in %s, %d\n", __FILE__, __LINE__);
        }
    err = cudaMemcpy(thradius, tdradius, natom*sizeof(double), cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"cudaMemcpy failed in %s, %d\n", __FILE__, __LINE__);
        }
    return cudaSuccess;
    }
    
void gen_config( tpvec *tcon, double *tradius, tpbox *tbox, tpsets tsets )
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

    // 3. cal area of disks
    double sdisk = 0.0;
    for ( int i=0; i<tbox->natom; i++ )
        {
        sdisk += tradius[i]*tradius[i];
        }
    sdisk *= Pi;

    // 4. cal box.l from phi
    double vol   = sdisk / tbox->phi;
    double temp  = sqrt(vol);
    tbox->x      = temp;
    tbox->y      = temp;
    tbox->xinv   = 1.0 / temp;
    tbox->yinv   = 1.0 / temp;
    tbox->strain = 0.0;

    // 5. give a random config
    for ( int i=0; i<tbox->natom; i++ )
        {
        tcon[i].x = ( (double)rand()/RAND_MAX - 0.5 ) * tbox->x;
        tcon[i].y = ( (double)rand()/RAND_MAX - 0.5 ) * tbox->y;
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

