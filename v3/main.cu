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
    cudaSetDevice(3);

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

    hycon_t *hycon;
    check_cuda( cudaMallocManaged( &hycon, sizeof(hycon_t) ) );
    calc_hypercon_args( hycon, box ); hycon->first_time = 1;
    printf("nubmer of total cells are %d\n",hycon->nblocks);
    check_cuda( cudaMallocManaged( &hycon->extraflag,   hycon->nblocks*sizeof(int  ) ) );
    check_cuda( cudaMallocManaged( &hycon->cnatom,      hycon->nblocks*sizeof(int  ) ) );
    check_cuda( cudaMallocManaged( &hycon->neighb,   26*hycon->nblocks*sizeof(int  ) ) );
    check_cuda( cudaMallocManaged( &hycon->tag   , msoc*hycon->nblocks*sizeof(int  ) ) );
    check_cuda( cudaMallocManaged( &hycon->radius, msoc*hycon->nblocks*sizeof(double) ) );
    check_cuda( cudaMallocManaged( &hycon->r,      msoc*hycon->nblocks*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &hycon->v,      msoc*hycon->nblocks*sizeof(vec_t)  ) );
    check_cuda( cudaMallocManaged( &hycon->f,      msoc*hycon->nblocks*sizeof(vec_t)  ) );
    map( hycon );
    printf("Start mapping from normal configuration into hyperconfiguration\n");//debug
    gpu_make_hypercon( hycon, con, radius, box);

    FILE *fptr= fopen("i_con.dat", "w+");
    write_config( fptr, con, radius, &box );
    fclose(fptr);

    //for ( int tep = 0; tep < 1000; tep++ )
    gpu_calc_force( hycon, &press, box );
    printf("press is %26.16e\n", press);

    for ( int i = 0; i < box.natom; i++ ) {
        con[i].x = 0.0;
        con[i].y = 0.0;
        con[i].z = 0.0;
        }
    gpu_map_hypercon_con( hycon, con, conv, conf, radius);

    fptr= fopen("mapped_con.dat", "w+");
    write_config( fptr, con, radius, &box );
    fclose(fptr);

    fptr= fopen("i_conf.dat", "w+");
    write_config( fptr, conf, radius, &box );
    fclose(fptr);

    return 0;
    }
