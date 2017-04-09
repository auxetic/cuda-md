__global__ void kernel_hypercon_zero_v( tpblock *thypercon )
    {
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bid = bidx + bidy * gridDim.x;

    int tid = threadIdx.x;

    __shared__ bnatom;
    if ( tid == 0 )
        {
        bnatom = hypercon[bid].bnatom;
        }
    __syncthreads();
    
    if ( tid < bnatom )
        {
        hypercon[bid].vx = 0.0;
        hypercon[bid].vy = 0.0;
        }

cudaError_t gpu_hypercon_zero_v( tpblock *thypercon, tpbox tbox )
    {
    dim3 grid( tbox.nblockx, tbox.nblocky, 1 );
    dim3 threads( maxn_of_block, 1, 1 );

    kernel_hypercon_zero_v<<< grid, threads >>>( thypercon );

    return cudaSucess;
    }

    
__global__ void kernel_hypercon_zero_f( tpblock *thypercon )
    {
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bid = bidx + bidy * gridDim.x;

    int tid = threadIdx.x;

    __shared__ bnatom;
    if ( tid == 0 )
        {
        bnatom = hypercon[bid].bnatom;
        }
    __syncthreads();
    
    if ( tid < bnatom )
        {
        hypercon[bid].fx = 0.0;
        hypercon[bid].fy = 0.0;
        }

cudaError_t gpu_hypercon_zero_f( tpblock *thypercon, tpbox tbox )
    {
    dim3 grid( tbox.nblockx, tbox.nblocky, 1 );
    dim3 threads( maxn_of_block, 1, 1 );

    kernel_hypercon_zero_f<<< grid, threads >>>( thypercon );

    return cudaSucess;
    }


__global__ void kernel_update_vr( tpblock *thypercon, double dt, double hfdt )
    {
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bid = bidx + bidy * gridDim.x;

    int tid = threadIdx.x;

    __shared__ double rx[maxn_of_block];
    __shared__ double ry[maxn_of_block];
    __shared__ double vx[maxn_of_block];
    __shared__ double vy[maxn_of_block];
    __shared__ double fx[maxn_of_block];
    __shared__ double fy[maxn_of_block];
    
    __shared__ bnatom;
    if ( tid == 0 )
        {
        bnatom = thypercon[bid].bnatom;
        }
    
    if ( tid < maxn_of_block )
        {
        rx[tid] = thypercon[tid].rx[tid];
        ry[tid] = thypercon[tid].ry[tid];
        vx[tid] = thypercon[tid].vx[tid];
        vy[tid] = thypercon[tid].vy[tid];
        fx[tid] = thypercon[tid].fx[tid];
        fy[tid] = thypercon[tid].fy[tid];
        }
    __syncthreads();

    if ( tid < bnatom )
        {
        vx[tid] += fx[tid] * hfdt;
        vy[tid] += fy[tid] * hfdt;
        rx[tid] += vx[tid] * dt;
        ry[tid] += vy[tid] * dt;
        hypercon[bid].rx[tid] = rx[tid];
        hypercon[bid].ry[tid] = ry[tid];
        hypercon[bid].vx[tid] = vx[tid];
        hypercon[bid].vy[tid] = vy[tid];
        }
    }
    
cudaError_t gpu_update_vr( tpblock *thypercon, tpmdargs tmdargs )
    {
    dim3 grid( tbox.nblockx, tbox.nblocky, 1 );
    dim3 threads( maxn_of_block, 1, 1 );

    kernel_update_vr<<< grid, threads >>>( thypercon, tmdargs.dt, tmdargs.hfdt );

    return cudaSucess;
    }

    
__global__ void kernel_update_v( tpblock *thypercon, double hfdt )
    {
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bid = bidx + bidy * gridDim.x;

    int tid = threadIdx.x;

    __shared__ double vx[maxn_of_block];
    __shared__ double vy[maxn_of_block];
    __shared__ double fx[maxn_of_block];
    __shared__ double fy[maxn_of_block];
    
    __shared__ bnatom;
    if ( tid == 0 )
        {
        bnatom = thypercon[bid].bnatom;
        }
    
    if ( tid < maxn_of_block )
        {
        vx[tid] = thypercon[tid].vx[tid];
        vy[tid] = thypercon[tid].vy[tid];
        fx[tid] = thypercon[tid].fx[tid];
        fy[tid] = thypercon[tid].fy[tid];
        }
    __syncthreads();

    if ( tid < bnatom )
        {
        vx[tid] += fx[tid] * hfdt;
        vy[tid] += fy[tid] * hfdt;
        hypercon[bid].vx[tid] = vx[tid];
        hypercon[bid].vy[tid] = vy[tid];
        }
    }

cudaError_t gpu_update_v( tpblock *thypercon, tpbox tbox, tpmdargs tmdargs )
    {
    dim3 grid( tbox.nblockx, tbox.nblocky, 1 );
    dim3 threads( maxn_of_block, 1, 1 );

    kernel_update_v<<< grid, threads >>>( thypercon, tmdargs.dt, tmdargs.hfdt );

    return cudaSucess;
    }


__global__ void kernel_calc_fmax( tpblock *thypercon )
    {
    __shared__ double bfmax[maxn_of_block];

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bid = bidx + bidy * gridDim.x;

    int tid = threadIdx.x;

    fmax[tid] = 0.0;
    
    __shared__ bnatom;
    if ( tid == 0 )
        {
        bnatom = thypercon[bid].bnatom;
        }
    
    if ( tid < maxn_of_block )
        {
        double fx, fy;
        fx = thypercon[tid].fx[tid];
        fy = thypercon[tid].fy[tid];
        bfmax[tid] = fmax( fabs(fx), fabs(fy) );
        }

    __syncthreads();

    int j = blockIdx.x;
    j >>= 1;
    while( j > 0 )
        {
        if ( tid < j )
            {
            bfmax[tid] = fmax( bfmax[tid], bfmax[tid+j] );
            }
        __syncthreads();
        j >>= 1;
        }

    if ( tid == 0 ) 
        {
        atomicMax( &dfmax, bfmax[0] );
        }
    }

cudaError_t gpu_calc_fmax( tpblock *thypercon, tpbox tbox )
    {
    dim3 grid( tbox.nblockx, tbox.nblocky, 1 );
    dim3 threads( maxn_of_block, 1, 1 );

    kernel_calc_fmax<<< grid, threads >>>( thypercon );

    return cudaSucess;
    }


__global__ void kernel_calc_fire_para( tpblock *thypercon )
    {
    __shared__ bpower[maxn_of_block];
    __shared__ bfn[maxn_of_block];
    __shared__ bvn[maxn_of_block];

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bid = bidx + bidy * gridDim.x;

    int tid = threadIdx.x;
    
    bpower[tid] = 0.0;
    bfn[tid]    = 0.0;
    bvn[tid]    = 0.0;

    __shared__ bnatom;
    if ( tid == 0 )
        {
        bnatom = thypercon[bid].bnatom;
        }
    
    if ( tid < maxn_of_block )
        {
        double fx, fy, vx, vy;
        vx = thypercon[tid].vx[tid];
        vy = thypercon[tid].vy[tid];
        fx = thypercon[tid].fx[tid];
        fy = thypercon[tid].fy[tid];
        bpower[tid] = vx*fx + vy*fy;
        bfn[tid]    = fx*fx + fy*fy;
        bvn[tid]    = vx*vx + vy*vy;
        }

    __syncthreads();

    int j = blockIdx.x;
    j >>= 1;
    while( j > 0 )
        {
        if ( tid < j )
            {
            bpower[tid] += bpower[tid+j];
            bfn[tid]    += bfn[tid+j];
            bvn[tid]    += bvn[tid+j];
            }
        __syncthreads();
        j >>= 1;
        }

    if ( tid == 0 ) 
        {
        atomicAdd( &dpower , bpower[0] );
        atomicAdd( &dfn2   , bfn[0] );
        atomicAdd( &dvn2   , bvn[0] );
        }
    }

cudaError_t gpu_calc_fire_para( tpblock *thypercon, tpbox tbox )
    {
    dim3 grid( tbox.nblockx, tbox.nblocky, 1 );
    dim3 threads( maxn_of_block, 1, 1 );

    kernel_calc_fire_para
    kernel_calc_fire_para<<< grid, threads >>>( thypercon );

    return cudaSucess;
    }

__global__ void kernel_calc_force( tpblock *thypercon, tpbox tbox )
    {
    __shared__ brx[maxn_of_block];
    __shared__ bry[maxn_of_block];
    __shared__ bfx[maxn_of_block];
    __shared__ bfy[maxn_of_block];
    __shared__ bpx, bpy, bstress;

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bid = bidx + bidy * gridDim.x;

    int tid = threadIdx.x;

    if ( tid < maxn_of_block )
        {
        brx[tid] = thypercon[bid].rx[tid];
        bry[tid] = thypercon[bid].ry[tid];
        bfx[tid] = 0.0;
        bfy[tid] = 0.0;
        }

    __shared__ bnatom;
    if ( tid == 0 )
        {
        bnatom = thypercon[bid].bnatom;
        }
    __syncthreads();
    

    while ( tid < n1 )
        {
        int iatom, jatom;
        iatom = list[tid].i;
        jatom = list[tid].j;

        double dx, dy;
        dx = brx[jatom] - brx[iatom];
        dy = bry[jatom] - bry[iatom];

        double rij2 = sqrt( dx*dx + dy*dy );

        double dij;
        dij = brr[iatom] + brr[jatom];

        if ( rij2 < dij*dij )
            {
            rij = sqrt(rij2);
            dij = 1.0 / dij;

            double wij, fr;













