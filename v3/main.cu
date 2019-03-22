#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "system.h"
#include "config.h"
#include "mdfunc.h"

int main(void)
    {
    box_t  box;
    box.natom = 16384;
    box.phi   = 0.88;
    sets_t sets;
    sets.seed = 1;
    cudaSetDevice(0);

    double  press = 0.0;
    double  *radius = NULL;
    vec_t   *con    = NULL;
    vec_t   *conv   = NULL;
    vec_t   *conf   = NULL;
    check_cuda( cudaMallocManaged( &con,     box.natom*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &conv,    box.natom*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &conf,    box.natom*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &radius , box.natom*sizeof(double) ) );
    gen_config( con, radius, &box, sets );

    hycon_t hycon;
    calc_hypercon_args( &hycon, box );
    check_cuda( cudaMallocManaged( &hycon.blocks, hycon.args.nblocks*sizeof(cell_t) ) );
    printf("Start\n");//debug
    gpu_make_hypercon( hycon, con, radius, box);
    map( hycon );

    FILE *fptr= fopen("i_con.dat", "w+");
    write_config( fptr, con, radius, &box );
    fclose(fptr);

    for ( int tep = 0; tep < 1000; tep++ )
    gpu_calc_force( &hycon, &press, box );
    printf("%26.16e\n", press);

    gpu_map_hypercon_con( hycon, con, conv, conf, radius);


    fptr= fopen("i_conf.dat", "w+");
    write_config( fptr, conf, radius, &box );
    fclose(fptr);

    return 0;
    }
