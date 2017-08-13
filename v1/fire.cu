#include "fire.h"

// fire constant parameter
#define fire_const_fmax    1.e-11
#define fire_const_dt0     3.e-2
#define fire_const_dtmax   3.e-1
#define fire_const_beta0   1.e-1
#define fire_const_finc    1.1e0
#define fire_const_fdec    0.5e0
#define fire_const_fbeta   0.99e0
#define fire_const_nmin    5
#define fire_const_stepmax 1000000000
//#define fire_const_stepmax 20

// inner subroutines
cudaError_t gpu_calc_fire_para( tpvec *tdconv, tpvec *tdconf, tpbox tbox );
cudaError_t gpu_fire_modify_v( tpvec *tdconv, tpvec *tdconf, double tfire_onemb, double tfire_betamvndfn, tpbox tbox);
cudaError_t gpu_firecp_update_box( tpvec *tdcon, tpbox *firebox, double dt, double current_press, double target_press, double *lv, int tstep );

// variables
__managed__ double gsm_fire_vn2, gsm_fire_fn2, gsm_fire_power, gsm_fire_fmax;


void mini_fire_cv( tpvec *tcon, double *tradius, tpbox tbox )
    {
    // allocate arrays used in fire
    // 1. box
    tpbox firebox = tbox;
    // 2. config
    tpvec *firecon;
    firecon = (tpvec *)malloc( firebox.natom * sizeof(tpvec) );
    memcpy( firecon, tcon, firebox.natom*sizeof(tpvec) );

    // allocate arrays for gpu
    // 1. config, velocity, force
    tpvec *dcon, *dconv, *dconf;
    cudaMalloc( (void **)&dcon,  firebox.natom*sizeof(tpvec) );
    cudaMalloc( (void **)&dconv, firebox.natom*sizeof(tpvec) );
    cudaMalloc( (void **)&dconf, firebox.natom*sizeof(tpvec) );
    // 1.1
    //tpvec *hconv, *hconf;
    //hconv = (tpvec *)malloc( firebox.natom * sizeof(tpvec) );
    //hconf = (tpvec *)malloc( firebox.natom * sizeof(tpvec) );
    // 2 radius
    double *dradius;
    cudaMalloc( (void **)&dradius, firebox.natom*sizeof(double) );

    // copy data to gpu
    trans_con_to_gpu( dcon, dradius, firebox.natom, firecon, tradius );


    // init list
    // 1. host
    //tplist *hlist;
    //hlist = (tplist *)malloc(firebox.natom*sizeof(tplist));
    // 2. device
    tplist *dlist;
    cudaMalloc((void **)&dlist, firebox.natom*(sizeof(tplist)));
    // 3. hypercon // used for calc list
    // 3.1 hypercon set
    calc_nblocks( &hblockset, firebox );
    // 3.2 hypercon
    cudaMalloc((void **)&dblocks, hblockset.nblocks*sizeof(tpblock));

    // make list
    gpu_make_hypercon( dcon, dradius, firebox, dblocks, hblockset );
    gpu_make_list( dlist, dblocks, dcon, hblockset, firebox );
    //cudaMemcpy( hlist, dlist, firebox.natom*sizeof(tplist), cudaMemcpyDeviceToHost );

    // init fire
    gpu_zero_confv( dconv, firebox );
    gpu_zero_confv( dconf, firebox );

    // force
    double press;
    gpu_calc_force( dlist, dcon, dradius, dconf, &press, firebox );

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
            //printf( "making list \n" );
            gpu_make_hypercon( dcon, dradius, firebox, dblocks, hblockset );
            gpu_make_list( dlist, dblocks, dcon, hblockset, firebox );
            }

        // velocity verlet / integrate veclocity and config
        gpu_update_vr( dcon, dconv, dconf, firebox, dt );

        // calc force
        gpu_calc_force( dlist, dcon, dradius, dconf, &press, firebox );

        // velocity verlet / integrate velocity
        gpu_update_v( dconv, dconf, firebox, dt );

        // fire
        fire_count += 1;

        // save power, fn2, vn2 in global variabls
        gpu_calc_fire_para( dconv, dconf, firebox );

        if ( gsm_fire_fmax < fire_const_fmax )
            {
            printf( "done fmax = %e\n", gsm_fire_fmax );
            break;
            }

        double fire_onemb, fire_vndfn, fire_betamvndfn;
        fire_onemb = 1.0 - fire_beta;
        fire_vndfn = sqrt( gsm_fire_vn2 / gsm_fire_fn2 );
        fire_betamvndfn = fire_beta * fire_vndfn;

       //cudaMemcpy( hconv, dconv, firebox.natom*sizeof(tpvec), cudaMemcpyDeviceToHost );

        if ( gsm_fire_power >= 0.0 )
            {
            gpu_fire_modify_v( dconv, dconf, fire_onemb, fire_betamvndfn, firebox );
            }

        if ( gsm_fire_power >= 0.0 && fire_count > fire_const_nmin )
            {
            dt = fmin( dt*fire_const_finc, fire_const_dtmax );
            fire_beta *= fire_const_fbeta;
            }

        if ( gsm_fire_power < 0.0 )
            {
            fire_count = 0;
            dt *= fire_const_fdec;
            fire_beta = fire_const_beta0;
            gpu_zero_confv( dconv, firebox );
            }

        if ( step%100 == 0 )
            printf( "step=%0.6d, fmax=%16.6e, press=%16.6e, dt=%16.6e \n", step, gsm_fire_fmax, press, dt );

        }

    cudaMemcpy( firecon, dcon, firebox.natom*sizeof(tpvec), cudaMemcpyDeviceToHost );
    memcpy( tcon, firecon, firebox.natom*sizeof(tpvec) );

    cudaFree( dblocks );
    cudaFree( dlist );
    cudaFree( dcon  );
    cudaFree( dconv );
    cudaFree( dconf );
    cudaFree( dradius );

    }

void mini_fire_cp( tpvec *tcon, double *tradius, tpbox *tbox0, double target_press )
    {
    // allocate arrays used in fire
    // 1. box
    tpbox tbox = *tbox0;
    tpbox firebox = tbox;
    // 2. config
    tpvec *firecon;
    firecon = (tpvec *)malloc( firebox.natom * sizeof(tpvec) );
    memcpy( firecon, tcon, firebox.natom*sizeof(tpvec) );

    // allocate arrays for gpu
    // 1. config, velocity, force
    tpvec *dcon, *dconv, *dconf;
    cudaMalloc( (void **)&dcon,  firebox.natom*sizeof(tpvec) );
    cudaMalloc( (void **)&dconv, firebox.natom*sizeof(tpvec) );
    cudaMalloc( (void **)&dconf, firebox.natom*sizeof(tpvec) );
    // 1.1
    //tpvec *hconv, *hconf;
    //hconv = (tpvec *)malloc( firebox.natom * sizeof(tpvec) );
    //hconf = (tpvec *)malloc( firebox.natom * sizeof(tpvec) );
    // 2 radius
    double *dradius;
    cudaMalloc( (void **)&dradius, firebox.natom*sizeof(double) );

    // copy data to gpu
    trans_con_to_gpu( dcon, dradius, firebox.natom, firecon, tradius );

    // init list
    // 1. host
    //tplist *hlist;
    //hlist = (tplist *)malloc(firebox.natom*sizeof(tplist));
    // 2. device
    tplist *dlist;
    cudaMalloc((void **)&dlist, firebox.natom*(sizeof(tplist)));
    // 3. hypercon // used for calc list
    // 3.1 hypercon set
    calc_nblocks( &hblockset, firebox );
    // 3.2 hypercon
    cudaMalloc((void **)&dblocks, hblockset.nblocks*sizeof(tpblock));

    // make list
    gpu_make_hypercon( dcon, dradius, firebox, dblocks, hblockset );
    gpu_make_list( dlist, dblocks, dcon, hblockset, firebox );
    //cudaMemcpy( hlist, dlist, firebox.natom*sizeof(tplist), cudaMemcpyDeviceToHost );

    // init fire
    gpu_zero_confv( dconv, firebox );
    gpu_zero_confv( dconf, firebox );

    // force
    double press;
    gpu_calc_force( dlist, dcon, dradius, dconf, &press, firebox );

    double dt, fire_beta;
    dt = fire_const_dt0;
    fire_beta = fire_const_beta0;
    int fire_count;
    fire_count = 0;
    double lv = 0.0;

    // main section
    for ( int step = 1; step <= fire_const_stepmax; step++ )
        {
        // check and make list
        if ( gpu_check_list( dcon, firebox, dlist ) )
            {
            //printf( "making list \n" );
            recalc_nblocks( &hblockset, firebox );
            gpu_make_hypercon( dcon, dradius, firebox, dblocks, hblockset );
            gpu_make_list( dlist, dblocks, dcon, hblockset, firebox );
            }

        // velocity verlet / integrate veclocity and config
        gpu_update_vr( dcon, dconv, dconf, firebox, dt );
        gpu_firecp_update_box( dcon, &firebox, dt, press, target_press, &lv, step );

        // calc force
        gpu_calc_force( dlist, dcon, dradius, dconf, &press, firebox );

        // velocity verlet / integrate velocity
        gpu_update_v( dconv, dconf, firebox, dt );
        double lf;
        lf = ( press - target_press ) * target_press;
        lv += lf * dt * 0.5;

        // fire
        fire_count += 1;

        // save power, fn2, vn2 in global variabls
        gpu_calc_fire_para( dconv, dconf, firebox );

        if ( gsm_fire_fmax < fire_const_fmax && fabs( press - target_press ) < fire_const_fmax )
            {
            printf( "done fmax = %e\n", gsm_fire_fmax );
            break;
            }

        double fire_onemb, fire_vndfn, fire_betamvndfn;
        fire_onemb = 1.0 - fire_beta;
        fire_vndfn = sqrt( gsm_fire_vn2 / gsm_fire_fn2 );
        fire_betamvndfn = fire_beta * fire_vndfn;

       //cudaMemcpy( hconv, dconv, firebox.natom*sizeof(tpvec), cudaMemcpyDeviceToHost );
        if ( lf * lv <= 0.0 )
            lv = 0.0;

        if ( gsm_fire_power >= 0.0 )
            {
            gpu_fire_modify_v( dconv, dconf, fire_onemb, fire_betamvndfn, firebox );
            }

        if ( gsm_fire_power >= 0.0 && fire_count > fire_const_nmin )
            {
            dt = fmin( dt*fire_const_finc, fire_const_dtmax );
            fire_beta *= fire_const_fbeta;
            }

        if ( gsm_fire_power < 0.0 )
            {
            fire_count = 0;
            dt *= fire_const_fdec;
            fire_beta = fire_const_beta0;
            gpu_zero_confv( dconv, firebox );
            }

        if ( step%200 == 0 )
            printf( "step=%0.6d, f=%16.6e, p=%26.16e, l=%26.16e, dt=%16.6e \n", step, gsm_fire_fmax, press, firebox.x, dt );

        }

    cudaMemcpy( firecon, dcon, firebox.natom*sizeof(tpvec), cudaMemcpyDeviceToHost );
    memcpy( tcon, firecon, firebox.natom*sizeof(tpvec) );

    *tbox0 = firebox;

    cudaFree( dblocks );
    cudaFree( dlist );
    cudaFree( dcon  );
    cudaFree( dconv );
    cudaFree( dconf );
    cudaFree( dradius );

    }

__global__ void kernel_calc_fire_para( tpvec *tdconv, tpvec *tdconf, int natom )
    {
    __shared__ float sm_vn2[SHARE_BLOCK_SIZE];
    __shared__ float sm_fn2[SHARE_BLOCK_SIZE];
    __shared__ float sm_power[SHARE_BLOCK_SIZE];
    __shared__ float sm_fmax[SHARE_BLOCK_SIZE];
    // ^^ 4 * 4 * 256 = 8192

    const int tid = threadIdx.x;
    const int i   = threadIdx.x + blockIdx.x * blockDim.x;

    sm_vn2[tid]   = 0.0;
    sm_fn2[tid]   = 0.0;
    sm_power[tid] = 0.0;
    sm_fmax[tid]  = 0.0;

    if ( i < natom )
        {
        float fx, fy, vx, vy;
        fx = tdconf[i].x;
        fy = tdconf[i].y;
        vx = tdconv[i].x;
        vy = tdconv[i].y;
        sm_vn2[tid]   = vx*vx + vy*vy;
        sm_fn2[tid]   = fx*fx + fy*fy;
        sm_power[tid] = vx*fx + vy*fy;
        sm_fmax[tid]  = fmax( fabs(fx), fabs(fy) );
        }

    __syncthreads();

    int j = SHARE_BLOCK_SIZE;
    j >>= 1;
    while ( j != 0 )
        {
        if ( tid < j )
            {
            sm_vn2[tid]   += sm_vn2[tid+j];
            sm_fn2[tid]   += sm_fn2[tid+j];
            sm_power[tid] += sm_power[tid+j];
            sm_fmax[tid]   = fmax( sm_fmax[tid], sm_fmax[tid+j] );
            }
        __syncthreads();
        j >>= 1;
        }

    if ( tid == 0 )
        {
        atomicAdd( &gsm_fire_vn2   , (double)sm_vn2[0]   );
        atomicAdd( &gsm_fire_fn2   , (double)sm_fn2[0]   );
        atomicAdd( &gsm_fire_power , (double)sm_power[0] );
        atomicMax( &gsm_fire_fmax  , (double)sm_fmax[0]  );
        }
    }

cudaError_t gpu_calc_fire_para( tpvec *tdconv, tpvec *tdconf, tpbox tbox )
    {
    const int block_size = 1024;
    const int natom = tbox.natom;

    gsm_fire_power = 0.0;
    gsm_fire_fn2   = 0.0;
    gsm_fire_vn2   = 0.0;
    gsm_fire_fmax  = 0.0;

    dim3 grids( ( natom / block_size )+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_calc_fire_para <<< grids, threads >>> ( tdconv, tdconf, natom );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        exit(-1);
        }

    return err;
    }

__global__ void kernel_fire_modify_v( tpvec *tdconv, tpvec *tdconf, int natom, double fire_onemb, double fire_betamvndfn )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < natom )
        {
        double v, f;

        v = tdconv[i].x;
        f = tdconf[i].x;
        v = fire_onemb * v + fire_betamvndfn * f;
        tdconv[i].x = v;

        v = tdconv[i].y;
        f = tdconf[i].y;
        v = fire_onemb * v + fire_betamvndfn * f;
        tdconv[i].y = v;
        }
    }

cudaError_t gpu_fire_modify_v( tpvec *tdconv, tpvec *tdconf, double tfire_onemb, double tfire_betamvndfn, tpbox tbox)
    {
    const int block_size = 256;

    int natom = tbox.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_fire_modify_v <<< grids, threads >>> ( tdconv, tdconf, natom, tfire_onemb, tfire_betamvndfn );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        exit(-1);
        }

    return err;
    }

__global__ void kernel_firecp_update_box( tpvec *tdcon, double tscale, int natom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= natom )
        return;

    double rx = tdcon[i].x;
    double ry = tdcon[i].y;
    rx *= tscale;
    ry *= tscale;
    tdcon[i].x = rx;
    tdcon[i].y = ry;

    }

cudaError_t gpu_firecp_update_box( tpvec *tdcon, tpbox *firebox, double dt, double current_press, double target_press, double *lv, int tstep )
    {
    const int block_size = 1024;
    const int natom = firebox->natom;

    double l = firebox->x;
    double lf = ( current_press - target_press ) * target_press ;

    l += *lv * dt + lf * dt * dt * 0.5;
    *lv += lf * dt * 0.5;

    double scale = l / firebox->x;
    firebox->x = l;
    firebox->y = l;
    firebox->xinv = 1.0 / l;
    firebox->yinv = 1.0 / l;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );

    kernel_firecp_update_box<<< grids, threads >>>( tdcon, scale, natom );

    cudaError_t err;
    err = cudaDeviceSynchronize();

    if ( err != cudaSuccess )
        {
        fprintf(stderr, "cudaDeviceSync failed, %s, %d, err = %d\n", __FILE__, __LINE__, err);
        exit(-1);
        }

    return err;
    }
