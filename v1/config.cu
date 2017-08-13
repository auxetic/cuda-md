#include "config.h"

// variables define
// host
tpvec *con;
double *radius;
// device
tpvec  *dcon;
double *dradius;


// allocate memory space of config
void alloc_con( tpvec **thcon, double **thradius, int tnatom )
    {
    *thcon    = (tpvec *)malloc(tnatom*sizeof(tpvec));
    *thradius = (double*)malloc(tnatom*sizeof(double));
    }

// allocate memory space of config
cudaError_t device_alloc_con( tpvec **tdcon, double **tdradius, int tnatom )
    {
    cudaError_t err;
    err = cudaMalloc( (void **)tdcon, tnatom*sizeof(tpvec) );
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"Malloc failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
        exit(-1);
        }
    err = cudaMalloc( (void **)tdradius , tnatom*sizeof(double) );
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"Malloc failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
        exit(-1);
        }
    return cudaSuccess;
    }

// copy config from host to device
cudaError_t trans_con_to_gpu( tpvec *tdcon, double *tdradius, int tnatom,
                              tpvec *thcon, double *thradius )
    {
    cudaError_t err;
    err = cudaMemcpy(tdcon, thcon, tnatom*sizeof(tpvec), cudaMemcpyHostToDevice);
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"cudaMemcpy failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
        exit(-1);
        }
    err = cudaMemcpy(tdradius, thradius, tnatom*sizeof(double), cudaMemcpyHostToDevice);
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"cudaMemcpy failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
        exit(-1);
        }
    return cudaSuccess;
    }

// copy config from device to host
cudaError_t trans_con_to_host( tpvec *thcon, double *thradius, int tnatom,
                               tpvec *tdcon, double *tdradius )
    {
    cudaError_t err;
    err = cudaMemcpy(thcon, tdcon, tnatom*sizeof(tpvec), cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"cudaMemcpy failed in %s, %d\n", __FILE__, __LINE__);
        exit(-1);
        }
    err = cudaMemcpy(thradius, tdradius, tnatom*sizeof(double), cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess )
        {
        fprintf(stderr,"cudaMemcpy failed in %s, %d\n", __FILE__, __LINE__);
        exit(-1);
        }
    return cudaSuccess;
    }

// generate random config on host
void gen_config( tpvec *thcon, double *thradius, tpbox *tbox, tpsets tsets )
    {
    // 1. intiate random number generator;
    srand(tsets.seed);

    // 2. set atom radius
    for ( int i=0; i<tbox->natom; i++ )
        {
        if ( i < tbox->natom/2 )
            thradius[i] = 0.5;
        else
            thradius[i] = 0.5 * ratio;
        }

    // 3. calc area of disks
    double sdisk = 0.0;
    for ( int i=0; i<tbox->natom; i++ )
        {
        sdisk += thradius[i]*thradius[i];
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
        thcon[i].x = ( (double)rand()/RAND_MAX - 0.5 ) * tbox->x;
        thcon[i].y = ( (double)rand()/RAND_MAX - 0.5 ) * tbox->y;
        }
    }

// read config
void read_config( FILE *fio, tpvec *thcon, double *thradius, tpbox *tbox )
    {
    int natom;
    fscanf(fio, "%d", &natom);
    tbox->natom = natom;
    double len;
    fscanf(fio, "%le", &len);
    tbox->x = len;
    tbox->y = len;
    tbox->xinv = 1e0/len;
    tbox->yinv = 1e0/len;
    tbox->strain = 0e0;

    for ( int i=0; i<natom; i++ )
        {
        double x, y, r;
        fscanf(fio, "%le", &x);
        fscanf(fio, "%le", &y);
        fscanf(fio, "%le", &r);
        thcon[i].x  = x*len;
        thcon[i].y  = y*len;
        thradius[i] = r*len;
        }
    }


// move all atoms to central box
void trim_config( tpvec *thcon, tpbox tbox )
    {
    for ( int i=0; i<tbox.natom; i++ )
        {
        double cory;
        cory = round( thcon[i].y * tbox.yinv );
        thcon[i].x -= cory * tbox.y * tbox.strain;

        thcon[i].x -= round( thcon[i].x * tbox.xinv ) * tbox.x;
        //thcon[i].y -= round( thcon[i].y * tbox.yinv ) * tbox.y;
        thcon[i].y -= cory * tbox.y;
        }
    }

