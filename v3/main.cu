#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "system.h"
#include "config.h"

int main(void)
    {
    box_t  box;
    box.natom = 128;
    box.phi   = 0.88;
    sets_t sets;
    sets.seed = 1;
    cudaSetDevice(0);

    double  *radius = NULL;
    vec_t   *con    = NULL;
    check_cuda( cudaMallocManaged( &con,     box.natom*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &radius , box.natom*sizeof(double) ) );
    gen_config( con, radius, &box, sets );

    hycon_t hycon;
    calc_hypercon_args( &hycon, box );
    check_cuda( cudaMallocManaged( &hycon.oneblocks, hycon.args.nblocks*sizeof(cell_t) ) );
    printf("h\n");//debug
    gpu_make_hypercon( hycon, con, radius, box);
    map( hycon );

    return 0;
    }
