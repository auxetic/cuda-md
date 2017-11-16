#include "config.h"

// variables define
// host
tpvec  *con;
double *radius;
// device
tpvec  *dcon;
double *dradius;


// allocate memory space of config
void alloc_con( tpvec **tcon, double **tradius, int tnatom )
    {
    *tcon    = (tpvec *)malloc( tnatom*sizeof(tpvec)  );
    *tradius = (double*)malloc( tnatom*sizeof(double) );
    }

// allocate memory space of config
cudaError_t device_alloc_con( tpvec **tcon, double **tradius, int tnatom )
    {
    check_cuda( cudaMallocManaged( &tcon,     tnatom*sizeof(tpvec)  ) );
    check_cuda( cudaMallocManaged( &tradius , tnatom*sizeof(double) ) );
    return cudaSuccess;
    }

// generate random config on host
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

    // 3. calc area of disks
    double sdisk = 0.0;
    for ( int i=0; i<tbox->natom; i++ )
        {
        sdisk += tradius[i]*tradius[i];
        }
    sdisk *= Pi;

    // 4. cal box.l from phi
    double vol     = sdisk / tbox->phi;
    double lx      = sqrt(vol);
    tbox->len.x    = lx;
    tbox->len.y    = lx;
    tbox->leninv.x = 1.0 / lx;
    tbox->leninv.y = 1.0 / lx;
    tbox->strain   = 0.0;

    // 5. give a random config
    for ( int i=0; i<tbox->natom; i++ )
        {
        tcon[i].x = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        tcon[i].y = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        }
    }

// read config
void read_config( FILE *tfio, tpvec *tcon, double *tradius, tpbox *tbox )
    {
    int natom;
    fscanf(tfio, "%d", &natom);
    tbox->natom = natom;
    double lx;
    fscanf(tfio, "%le", &lx);
    tbox->len.x = lx;
    tbox->len.y = lx;
    tbox->leninv.x = 1e0/lx;
    tbox->leninv.y = 1e0/lx;
    tbox->strain = 0e0;

    for ( int i=0; i<natom; i++ )
        {
        double x, y, r;
        fscanf(tfio, "%le", &x);
        fscanf(tfio, "%le", &y);
        fscanf(tfio, "%le", &r);
        tcon[i].x  = x*lx;
        tcon[i].y  = y*lx;
        tradius[i] = r*lx;
        }
    }

// move all atoms to central box
void trim_config( tpvec *tcon, tpbox tbox )
    {
    double lx = tbox.len.x;
    double ly = tbox.len.y;
    for ( int i=0; i<tbox.natom; i++ )
        {
        double cory;
        cory = round( tcon[i].y / ly );
        tcon[i].x -= cory * ly * tbox.strain;

        tcon[i].x -= round( tcon[i].x / lx ) * lx;
        tcon[i].y -= cory * ly;
        }
    }

__global__ void kernel_trim_config( tpvec *tcon, int tnatom, double lx, double ly )
    {
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < tnatom )
        {
        double x, y;
        x = tcon[i].x;
        y = tcon[i].y;

        x -= round( x/lx ) * lx;
        y -= round( y/ly ) * ly;

        tcon[i].x = x;
        tcon[i].y = y;
        }
    }

cudaError_t gpu_trim_config( tpvec *tcon, tpbox tbox )
    {
    const int    natom = tbox.natom;
    const double lx    = tbox.len.x;
    const double ly    = tbox.len.y;

    const int    block_size = 256;
    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_trim_config<<< grids, threads >>>( tcon, natom, lx, ly );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

//// copy config from host to device
    //cudaError_t trans_con_to_gpu( tpvec *tdcon, double *tdradius, int tnatom,
    //                              tpvec *thcon, double *thradius )
    //    {
    //    cudaError_t err;
    //    err = cudaMemcpy(tdcon, thcon, tnatom*sizeof(tpvec), cudaMemcpyHostToDevice);
    //    if ( err != cudaSuccess )
    //        {
    //        fprintf(stderr,"cudaMemcpy failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
    //        exit(-1);
    //        }
    //    err = cudaMemcpy(tdradius, thradius, tnatom*sizeof(double), cudaMemcpyHostToDevice);
    //    if ( err != cudaSuccess )
    //        {
    //        fprintf(stderr,"cudaMemcpy failed in %s, %d, err=%d\n", __FILE__, __LINE__, err);
    //        exit(-1);
    //        }
    //    return cudaSuccess;
    //    }

//// copy config from device to host
    //cudaError_t trans_con_to_host( tpvec *thcon, double *thradius, int tnatom,
    //                               tpvec *tdcon, double *tdradius )
    //    {
    //    cudaError_t err;
    //    err = cudaMemcpy(thcon, tdcon, tnatom*sizeof(tpvec), cudaMemcpyDeviceToHost);
    //    if ( err != cudaSuccess )
    //        {
    //        fprintf(stderr,"cudaMemcpy failed in %s, %d\n", __FILE__, __LINE__);
    //        exit(-1);
    //        }
    //    err = cudaMemcpy(thradius, tdradius, tnatom*sizeof(double), cudaMemcpyDeviceToHost);
    //    if ( err != cudaSuccess )
    //        {
    //        fprintf(stderr,"cudaMemcpy failed in %s, %d\n", __FILE__, __LINE__);
    //        exit(-1);
    //        }
    //    return cudaSuccess;
    //    }
