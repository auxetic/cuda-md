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
//#define fire_const_stepmax 200

// inner subroutines
cudaError_t gpu_calc_fire_para( vec_t *tdconv, vec_t *tdconf, box_t tbox );
cudaError_t gpu_fire_modify_v( vec_t *tdconv, vec_t *tdconf, double tfire_onemb, double tfire_betamvndfn, box_t tbox);
cudaError_t gpu_firecp_update_box( vec_t *tdcon, box_t *firebox, double dt, double current_press, double target_press, double *lv, int tstep );

// variables
__managed__ double g_fire_vn2, g_fire_fn2, g_fire_power, g_fire_fmax;

void mini_fire_cv( vec_t *tcon, double *tradius, box_t tbox )
    {
    // allocate arrays used in fire
    // 1. box
    box_t firebox = tbox;

    // 2. config
    vec_t *hdcon, *hdconv, *hdconf; double *hdradius;
    cudaMallocManaged( &hdcon,    firebox.natom*sizeof(vec_t)  );
    cudaMallocManaged( &hdconv,   firebox.natom*sizeof(vec_t)  );
    cudaMallocManaged( &hdconf,   firebox.natom*sizeof(vec_t)  );
    cudaMallocManaged( &hdradius, firebox.natom*sizeof(double) );
    memcpy( hdcon,    tcon,    firebox.natom*sizeof(vec_t)  );
    memcpy( hdradius, tradius, firebox.natom*sizeof(double) );

    // init list
    list_t hdlist;
    hdlist.natom = firebox.natom;
    cudaMallocManaged( &(hdlist.onelists), hdlist.natom*sizeof(onelist_t) );
    // hypercon
    blocks_t hdblock;
    calc_nblocks( &hdblock, firebox );
    cudaMallocManaged( &(hdblock.oneblocks), hdblock.args.nblocks*sizeof(oneblock_t) );
    // make list
    printf( "making hypercon \n" );
    gpu_make_hypercon( hdblock, hdcon, hdradius, firebox );
    printf( "making list \n" );
    gpu_make_list( hdlist, hdblock, hdcon, firebox );
    // test
    //list_t hlist;
    //hlist.natom = firebox.natom;
    //cudaMallocManaged( &(hlist.onelists), hlist.natom*sizeof(onelist_t) );
    ////cpu_make_list( hlist, hdcon, tradius, firebox );
    //gpu_make_list_fallback( hlist, hdcon, hdradius, firebox );

    // init fire
    gpu_zero_confv( hdconv, firebox );
    gpu_zero_confv( hdconf, firebox );

    // force
    double press;
    gpu_calc_force( hdconf, hdlist, hdcon, hdradius, &press, firebox );

    double dt, fire_beta;
    dt = fire_const_dt0;
    fire_beta = fire_const_beta0;
    int fire_count;
    fire_count = 0;

    // main section
    for ( int step = 1; step <= fire_const_stepmax; step++ )
        {
        // check and make list
        if ( gpu_check_list( hdlist, hdcon, firebox ) )
            {
            printf( "making list \n" );
            gpu_trim_config( hdcon, firebox );
            gpu_make_hypercon( hdblock, hdcon, hdradius, firebox );
            gpu_make_list( hdlist, hdblock, hdcon, firebox );
            }

        // velocity verlet / integrate veclocity and config
        gpu_update_vr( hdcon, hdconv, hdconf, firebox, dt );

        // calc force
        gpu_calc_force( hdconf, hdlist, hdcon, hdradius, &press, firebox );

        // velocity verlet / integrate velocity
        gpu_update_v( hdconv, hdconf, firebox, dt );

        // fire
        fire_count += 1;

        // save power, fn2, vn2 in global variabls
        gpu_calc_fire_para( hdconv, hdconf, firebox );

        double ffmax = gpu_calc_fmax( hdconf, firebox );

        if ( g_fire_fmax < fire_const_fmax )
            {
            printf( "done fmax = %e\n", g_fire_fmax );
            break;
            }

        double fire_onemb, fire_vndfn, fire_betamvndfn;
        fire_onemb = 1.0 - fire_beta;
        fire_vndfn = sqrt( g_fire_vn2 / g_fire_fn2 );
        fire_betamvndfn = fire_beta * fire_vndfn;

        if ( g_fire_power >= 0.0 )
            {
            gpu_fire_modify_v( hdconv, hdconf, fire_onemb, fire_betamvndfn, firebox );
            }

        if ( g_fire_power >= 0.0 && fire_count > fire_const_nmin )
            {
            dt = fmin( dt*fire_const_finc, fire_const_dtmax );
            fire_beta *= fire_const_fbeta;
            }

        if ( g_fire_power < 0.0 )
            {
            fire_count = 0;
            dt *= fire_const_fdec;
            fire_beta = fire_const_beta0;
            gpu_zero_confv( hdconv, firebox );
            }

        if ( step%100 == 0 )
            printf( "step=%0.6d, fmax=%16.6e, press=%16.6e, dt=%16.6e \n", step, g_fire_fmax, press, dt );

        }

    memcpy( tcon, hdcon, firebox.natom*sizeof(vec_t) );

    cudaFree( hdblock.oneblocks );
    cudaFree( hdlist.onelists );
    cudaFree( hdcon  );
    cudaFree( hdconv );
    cudaFree( hdconf );
    cudaFree( hdradius );
    }

void mini_fire_cp( vec_t *tcon, double *tradius, box_t *tbox0, double target_press )
    {
    // allocate arrays used in fire
    // 1. box
    box_t tbox = *tbox0;
    box_t firebox = tbox;
    // 2. config
    vec_t *hdcon, *hdconv, *hdconf; double *hdradius;
    cudaMallocManaged( &hdcon,    firebox.natom*sizeof(vec_t)  );
    cudaMallocManaged( &hdconv,   firebox.natom*sizeof(vec_t)  );
    cudaMallocManaged( &hdconf,   firebox.natom*sizeof(vec_t)  );
    cudaMallocManaged( &hdradius, firebox.natom*sizeof(double) );
    memcpy( hdcon,    tcon,    firebox.natom*sizeof(vec_t)  );
    memcpy( hdradius, tradius, firebox.natom*sizeof(double) );

    // init list
    list_t hdlist; //hlist;
    hdlist.natom = firebox.natom;
    cudaMallocManaged( &(hdlist.onelists), hdlist.natom*sizeof(onelist_t) );
    // hypercon
    blocks_t hdblock;
    calc_nblocks( &hdblock, firebox );
    cudaMallocManaged( &(hdblock.oneblocks), hdblock.args.nblocks*sizeof(oneblock_t) );
    // make list
    printf( "making hypercon \n" );
    gpu_make_hypercon( hdblock, hdcon, hdradius, firebox );
    printf( "making list \n" );
    gpu_make_list( hdlist, hdblock, hdcon, firebox );

    // init fire
    gpu_zero_confv( hdconv, firebox );
    gpu_zero_confv( hdconf, firebox );

    // force
    double press;
    gpu_calc_force( hdconf, hdlist, hdcon, hdradius, &press, firebox );

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
        if ( gpu_check_list( hdlist, hdcon, firebox ) )
            {
            printf( "making list \n" );
            gpu_trim_config( hdcon, firebox );
            gpu_make_hypercon( hdblock, hdcon, hdradius, firebox );
            gpu_make_list( hdlist, hdblock, hdcon, firebox );
            }

        // velocity verlet / integrate veclocity and config
        gpu_update_vr( hdcon, hdconv, hdconf, firebox, dt );
        gpu_firecp_update_box( hdcon, &firebox, dt, press, target_press, &lv, step );

        // calc force
        gpu_calc_force( hdconf, hdlist, hdcon, hdradius, &press, firebox );

        // velocity verlet / integrate velocity
        gpu_update_v( hdconv, hdconf, firebox, dt );
        double lf;
        lf = 0.0; //( press - target_press ) * target_press;
        lv += lf * dt * 0.5;

        // fire
        fire_count += 1;

        // save power, fn2, vn2 in global variabls
        gpu_calc_fire_para( hdconv, hdconf, firebox );

        if ( g_fire_fmax < fire_const_fmax && fabs( press - target_press ) < fire_const_fmax )
            {
            printf( "done fmax = %e\n", g_fire_fmax );
            break;
            }

        double fire_onemb, fire_vndfn, fire_betamvndfn;
        fire_onemb = 1.0 - fire_beta;
        fire_vndfn = sqrt( g_fire_vn2 / g_fire_fn2 );
        fire_betamvndfn = fire_beta * fire_vndfn;

       //cudaMemcpy( hconv, dconv, firebox.natom*sizeof(vec_t), cudaMemcpyDeviceToHost );
        if ( lf * lv <= 0.0 )
            lv = 0.0;

        if ( g_fire_power >= 0.0 )
            {
            gpu_fire_modify_v( hdconv, hdconf, fire_onemb, fire_betamvndfn, firebox );
            }

        if ( g_fire_power >= 0.0 && fire_count > fire_const_nmin )
            {
            dt = fmin( dt*fire_const_finc, fire_const_dtmax );
            fire_beta *= fire_const_fbeta;
            }

        if ( g_fire_power < 0.0 )
            {
            fire_count = 0;
            dt *= fire_const_fdec;
            fire_beta = fire_const_beta0;
            gpu_zero_confv( hdconv, firebox );
            }

        if ( step%100 == 0 )
            printf( "step=%0.6d, f=%16.6e, p=%26.16e, l=%26.16e, dt=%16.6e \n", step, g_fire_fmax, press, firebox.len.x, dt );

        }

    memcpy( tcon, hdcon, firebox.natom*sizeof(vec_t) );

    *tbox0 = firebox;

    cudaFree( hdblock.oneblocks );
    cudaFree( hdlist.onelists );
    cudaFree( hdcon  );
    cudaFree( hdconv );
    cudaFree( hdconf );
    cudaFree( hdradius );
    }

__global__ void kernel_calc_fire_para( vec_t *thdconv, vec_t *thdconf, int tnatom )
    {
    __shared__ double sm_vn2[SHARE_BLOCK_SIZE];
    __shared__ double sm_fn2[SHARE_BLOCK_SIZE];
    __shared__ double sm_power[SHARE_BLOCK_SIZE];
    __shared__ double sm_fmax[SHARE_BLOCK_SIZE];
    // ^^ 4 * 4 * 256 = 8192

    const int tid = threadIdx.x;
    const int i   = threadIdx.x + blockIdx.x * blockDim.x;

    sm_vn2[tid]   = 0.0;
    sm_fn2[tid]   = 0.0;
    sm_power[tid] = 0.0;
    sm_fmax[tid]  = 0.0;

    if ( i < tnatom )
        {
        double fx, fy, vx, vy;
        fx = thdconf[i].x;
        fy = thdconf[i].y;
        vx = thdconv[i].x;
        vy = thdconv[i].y;
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
        atomicAdd( &g_fire_vn2   , sm_vn2[0]   );
        atomicAdd( &g_fire_fn2   , sm_fn2[0]   );
        atomicAdd( &g_fire_power , sm_power[0] );
        atomicMax( &g_fire_fmax  , sm_fmax[0]  );
        }
    }

cudaError_t gpu_calc_fire_para( vec_t *thdconv, vec_t *thdconf, box_t tbox )
    {
    const int block_size = 512;
    const int natom = tbox.natom;

    g_fire_power = 0.0;
    g_fire_fn2   = 0.0;
    g_fire_vn2   = 0.0;
    g_fire_fmax  = 0.0;

    dim3 grids( ( natom / block_size )+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_calc_fire_para <<< grids, threads >>> ( thdconv, thdconf, natom );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

__global__ void kernel_fire_modify_v( vec_t *thdconv, vec_t *thdconf, int tnatom, double fire_onemb, double fire_betamvndfn )
    {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < tnatom )
        {
        double v, f;

        v = thdconv[i].x;
        f = thdconf[i].x;
        v = fire_onemb * v + fire_betamvndfn * f;
        thdconv[i].x = v;

        v = thdconv[i].y;
        f = thdconf[i].y;
        v = fire_onemb * v + fire_betamvndfn * f;
        thdconv[i].y = v;
        }
    }

cudaError_t gpu_fire_modify_v( vec_t *thdconv, vec_t *thdconf, double tfire_onemb, double tfire_betamvndfn, box_t tbox)
    {
    const int block_size = 256;

    int natom = tbox.natom;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_fire_modify_v <<< grids, threads >>> ( thdconv, thdconf, natom, tfire_onemb, tfire_betamvndfn );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }

__global__ void kernel_firecp_update_box( vec_t *thdcon, double tscale, int tnatom )
    {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= tnatom )
        return;

    double rx = thdcon[i].x;
    double ry = thdcon[i].y;
    rx *= tscale;
    ry *= tscale;
    thdcon[i].x = rx;
    thdcon[i].y = ry;
    }

cudaError_t gpu_firecp_update_box( vec_t *thdcon, box_t *firebox, double dt, double current_press, double target_press, double *lv, int tstep )
    {
    const int block_size = 1024;
    const int natom = firebox->natom;

    double l = firebox->len.x;
    double lf = ( current_press - target_press ) * target_press ;

    l += *lv * dt + lf * dt * dt * 0.5;
    *lv += lf * dt * 0.5;

    double scale = l / firebox->len.x;
    firebox->len.x = l;
    firebox->len.y = l;
    firebox->leninv.x = 1.0 / l;
    firebox->leninv.y = 1.0 / l;

    dim3 grids( (natom/block_size)+1, 1, 1 );
    dim3 threads( block_size, 1, 1 );
    kernel_firecp_update_box<<< grids, threads >>>( thdcon, scale, natom );
    check_cuda( cudaDeviceSynchronize() );

    return cudaSuccess;
    }
