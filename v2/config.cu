#include "config.h"

// variables define
// host
tpvec  *con;
double *radius;
// device
tpvec  *dcon;
double *dradius;


// allocate memory space of config
void alloc_con( tpvec **_con, double **_radius, int _natom )
    {
    *_con    = (tpvec *)malloc(_natom*sizeof(tpvec));
    *_radius = (double*)malloc(_natom*sizeof(double));
    }

// allocate memory space of config
cudaError_t device_alloc_con( tpvec **_con, double **_radius, int _natom )
    {
    check_cuda( cudaMallocManaged( &_con,     _natom*sizeof(tpvec)  ) );
    check_cuda( cudaMallocManaged( &_radius , _natom*sizeof(double) ) );
    return cudaSuccess;
    }

// generate random config on host
void gen_config( tpvec *_con, double *_radius, tpbox *_box, tpsets _sets )
    {
    // 1. intiate random number generator;
    srand(_sets.seed);

    // 2. set atom radius
    for ( int i=0; i<_box->natom; i++ )
        {
        if ( i < _box->natom/2 )
            _radius[i] = 0.5;
        else
            _radius[i] = 0.5 * ratio;
        }

    // 3. calc area of disks
    double sdisk = 0.0;
    for ( int i=0; i<_box->natom; i++ )
        {
        sdisk += _radius[i]*_radius[i];
        }
    sdisk *= Pi;

    // 4. cal box.l from phi
    double vol     = sdisk / _box->phi;
    double lx      = sqrt(vol);
    _box->len.x    = lx;
    _box->len.y    = lx;
    _box->leninv.x = 1.0 / lx;
    _box->leninv.y = 1.0 / lx;
    _box->strain   = 0.0;

    // 5. give a random config
    for ( int i=0; i<_box->natom; i++ )
        {
        _con[i].x = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        _con[i].y = ( (double)rand()/RAND_MAX - 0.5 ) * lx;
        }
    }

// read config
void read_config( FILE *_fio, tpvec *_con, double *_radius, tpbox *_box )
    {
    int natom;
    fscanf(_fio, "%d", &natom);
    _box->natom = natom;
    double lx;
    fscanf(_fio, "%le", &lx);
    _box->lx.x = lx;
    _box->lx.y = lx;
    _box->lxinv.x = 1e0/lx;
    _box->lxinv.y = 1e0/lx;
    _box->strain = 0e0;

    for ( int i=0; i<natom; i++ )
        {
        double x, y, r;
        fscanf(_fio, "%le", &x);
        fscanf(_fio, "%le", &y);
        fscanf(_fio, "%le", &r);
        _con[i].x  = x*lx;
        _con[i].y  = y*lx;
        _radius[i] = r*lx;
        }
    }


// move all atoms to central box
void trim_config( tpvec *_con, tpbox _box )
    {
    double lx = _box.len.x;
    double ly = _box.len.y;
    for ( int i=0; i<_box.natom; i++ )
        {
        double cory;
        cory = round( _con[i].y / ly );
        _con[i].x -= cory * ly * _box.strain;

        _con[i].x -= round( _con[i].x / lx ) * lx;
        _con[i].y -= cory * ly;
        }
    }

__global__ void kernel_trim_config( tpvec *_con, int _natom, double lx, double ly )
    {
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < _natom )
        {
        double x, y;
        x = _con[i].x;
        y = _con[i].y;

        x -= round( x/lx ) * lx;
        y -= round( y/ly ) * ly;

        _con[i].x = x;
        _con[i].y = y;
        }
    }

cudaError_t gpu_trim_config( tpvec *_con, tpbox _box )
    {
    const int    natom = _box.natom;
    const double lx    = _box.len.x;
    const double ly    = _box.len.y;

    const int    block_size = 256;
    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_trim_config<<< grids, threads >>>( _con, natom, lx, ly );

    check_cuda( cudaDeviceSync() );

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
