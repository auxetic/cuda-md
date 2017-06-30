#include "fire.h"

// fire constant parameter
#define fire_const_fmax    1.e-8
#define fire_const_dt0     3.e-2
#define fire_const_dtmax   3.e-1
#define fire_const_beta0   1.e-1
#define fire_const_finc    1.1e0
#define fire_const_fdec    0.5e0
#define fire_const_fbeta   0.99e0
#define fire_const_nmin    5 
#define fire_const_stepmax 5000000

// variables
__managed__ double mfire_vn2, mfire_fn2, mfire_power, mfire_fmax;


void mini_fire_cv( tpvec *tcon, double *tradius, tpbox tbox )
    {
    // allocate arrays used in fire
    // 1. box
    tpbox firebox = tbox;
    // 2. config
    tpvec *firecon;
    firecon = (tpvec *)malloc( firebox.natom * sizeof(tpvec) );
    memcpy( firecon, tcon, firebox.natom*sizeof(tpvec) );
    // 3. radius
    double *fireradius;
    fireradius = (double *)malloc(firebox.natom*sizeof(double));
    memcpy( fireradius, tradius, firebox.natom*sizeof(double) );

    // allocate arrays for gpu
    // 1. config, velocity, force
    tpvec *dcon, *dconv, *dconf;
    cudaMalloc( (void **)&dcon,  firebox.natom*sizeof(tpvec) );
    cudaMalloc( (void **)&dconv, firebox.natom*sizeof(tpvec) );
    cudaMalloc( (void **)&dconf, firebox.natom*sizeof(tpvec) );
    // 1.1
    tpvec *hconv;
    hconv = (tpvec *)malloc( firebox.natom * sizeof(tpvec) );
    // 2 radius
    double *dradius;
    cudaMalloc( (void **)&dradius, firebox.natom*sizeof(double) );

    // copy data to gpu
    //cudaMemcpy( dcon,    firecon,    firebox.natom*sizeof(tpvec),  cudaMemcpyHostToDevice );
    //cudaMemcpy( dradius, fireradius, firebox.natom*sizeof(double), cudaMemcpyHostToDevice );
    trans_con_to_gpu( firecon, fireradius, firebox.natom, dcon, dradius );


    // init list
    // 1. host
    tplist *hlist;
    hlist = (tplist *)malloc(firebox.natom*sizeof(tplist));
    // 2. device
    tplist *dlist;
    cudaMalloc((void **)&dlist, firebox.natom*(sizeof(tplist)));
    // 3. hypercon // used for calc list
    // 3.1 hypercon set
    tpblockset hyperconset;
    calc_nblocks( &hyperconset, firebox.x, firebox.y, firebox.natom );
    // 3.2 hypercon
    tpblock *hypercon;
    cudaMalloc((void **)&hypercon, hyperconset.nblocks*sizeof(tpblock));

    // make list
    gpu_make_hypercon( dcon, dradius, firebox, hypercon, hyperconset );
    gpu_make_list( hypercon, dcon, hyperconset, firebox, dlist );
    //cudaMemcpy( hlist, dlist, firebox.natom*sizeof(tplist), cudaMemcpyDeviceToHost );

    // force
    gpu_calc_force( dlist, dcon, dradius, dconf, firebox );

    // init fire
    gpu_zero_confv( dconv, firebox );

    double dt, fire_beta;
    dt = fire_const_dt0;
    fire_beta = fire_const_beta0;
    int fire_count;
    fire_count = 0;

    // main section
    for ( int step = 1; step <= fire_const_stepmax; step++ )
        {
        // check and make list
        if ( gpu_check_list( dcon, firebox, dlist ) )
            {
            printf( "making list \n" );
            gpu_make_hypercon( dcon, dradius, firebox, hypercon, hyperconset );
            gpu_make_list( hypercon, dcon, hyperconset, firebox, dlist );
            }
        
        // velocity verlet / integrate veclocity and config
        gpu_update_vr( dcon, dconv, dconf, firebox, dt );

        // calc force
        gpu_calc_force( dlist, dcon, dradius, dconf, firebox );

        // velocity verlet / integrate velocity
        gpu_update_v( dconv, dconf, firebox, dt );

        // fire
        fire_count += 1;

        // save power, fn2, vn2 in global variabls
        gpu_calc_fire_para( dconv, dconf, firebox );

        if ( mfire_fmax < fire_const_fmax )
            {
            printf( "done fmax = %e\n", mfire_fmax );
            break;
            }

        double fire_onemb, fire_vndfn, fire_betamvndfn;
        fire_onemb = 1.0 - fire_beta;
        fire_vndfn = sqrt( mfire_vn2 / mfire_fn2 );
        fire_betamvndfn = fire_beta * fire_vndfn;

       //cudaMemcpy( hconv, dconv, firebox.natom*sizeof(tpvec), cudaMemcpyDeviceToHost );

        if ( mfire_power >= 0.0 )
            {
            gpu_fire_modify_v( dconv, dconf, fire_onemb, fire_betamvndfn, firebox );
            }

        if ( mfire_power >= 0.0 && fire_count > fire_const_nmin )
            {
            dt = fmin( dt*fire_const_finc, fire_const_dtmax );
            fire_beta *= fire_const_fbeta;
            }

        if ( mfire_power < 0.0 ) 
            {
            fire_count = 0;
            dt *= fire_const_fdec;
            fire_beta = fire_const_beta0;
            gpu_zero_confv( dconv, firebox );
            }

        if ( step%100 == 0 )
            printf( "step=%0.6d, fmax=%16.6e, power=%16.6e, fn2=%16.6e, vn2=%16.6e, dt=%16.6e \n", step, mfire_fmax, mfire_power, mfire_fn2, mfire_vn2, dt );

        }

        cudaMemcpy( firecon, dcon, firebox.natom*sizeof(tpvec), cudaMemcpyDeviceToHost );
        memcpy( tcon, firecon, firebox.natom*sizeof(tpvec) );

        cudaFree( hypercon );
        cudaFree( dlist );
        cudaFree( dcon  );
        cudaFree( dconv );
        cudaFree( dconf );
        cudaFree( dradius );
    
    }

__global__ void kernel_calc_fire_para( tpvec *tdconv, tpvec *tdconf, int natom )
    {
    __shared__ float svn2[SHARE_BLOCK_SIZE];
    __shared__ float sfn2[SHARE_BLOCK_SIZE];
    __shared__ float spower[SHARE_BLOCK_SIZE];
    __shared__ float sfmax[SHARE_BLOCK_SIZE];
    // ^^ 4 * 4 * 256 = 8192

    const int tid = threadIdx.x;
    const int i   = threadIdx.x + blockIdx.x * blockDim.x;

    svn2[tid]   = 0.0;
    sfn2[tid]   = 0.0;
    spower[tid] = 0.0;
    sfmax[tid]  = 0.0;

    if ( i < natom )
        {
        float fx, fy, vx, vy;
        fx = tdconf[i].x; 
        fy = tdconf[i].y; 
        vx = tdconv[i].x; 
        vy = tdconv[i].y; 
        svn2[tid]   = vx*vx + vy*vy;
        sfn2[tid]   = fx*fx + fy*fy;
        spower[tid] = vx*fx + vy*fy;
        sfmax[tid]  = fmax( fabs(fx), fabs(fy) );
        }

    __syncthreads();

    int j = SHARE_BLOCK_SIZE;
    j >>= 1;
    while ( j != 0 )
        { 
        if ( tid < j )
            {
            svn2[tid]   += svn2[tid+j];
            sfn2[tid]   += sfn2[tid+j];
            spower[tid] += spower[tid+j];
            sfmax[tid]   = fmax( sfmax[tid], sfmax[tid+j] );
            }
        __syncthreads();
        j >>= 1;
        }

    if ( tid == 0 )
        {
        atomicAdd( &mfire_vn2   , (double)svn2[0]   );
        atomicAdd( &mfire_fn2   , (double)sfn2[0]   );
        atomicAdd( &mfire_power , (double)spower[0] );
        atomicMax( &mfire_fmax  , (double)sfmax[0]  );
        }
    }

cudaError_t gpu_calc_fire_para( tpvec *tdconv, tpvec *tdconf, tpbox tbox )
    {
    const int block_size = 1024;
    const int natom = tbox.natom;

    mfire_power = 0.0;
    mfire_fn2   = 0.0;
    mfire_vn2   = 0.0;
    mfire_fmax  = 0.0;

    dim3 grids( ceil( natom / block_size )+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_calc_fire_para <<< grids, threads >>> ( tdconv, tdconf, natom );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }

__global__ void kernel_fire_modify_v( tpvec *tdconv, tpvec *tdconf, int natom, double fire_onemb, double fire_betamvndfn )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        double vxi, vyi;
        vxi = tdconv[i].x;
        vyi = tdconv[i].y;

        double fxi, fyi;
        fxi = tdconf[i].x;
        fyi = tdconf[i].y;

        vxi = fire_onemb * vxi + fire_betamvndfn * fxi;
        vyi = fire_onemb * vyi + fire_betamvndfn * fyi;

        tdconv[i].x = vxi;
        tdconv[i].y = vyi;
        }
    }

cudaError_t gpu_fire_modify_v( tpvec *tdconv, tpvec *tdconf, double tfire_onemb, double tfire_betamvndfn, tpbox tbox)
    {
    const int block_size = 256;

    int natom = tbox.natom;

    dim3 grids( ceil(natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_fire_modify_v <<< grids, threads >>> ( tdconv, tdconf, natom, tfire_onemb, tfire_betamvndfn );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        }

    return err;
    }
