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
    box.natom = 1024;
    box.phi   = 0.88;
    sets_t sets;
    sets.seed = 1;
    cudaSetDevice(0);

    vec_t   *con;
    double  *radius;
    hycon_t *hycon;

    alloc_managed_con( con, radius, box.natom );
    gen_config( con, radius, box, sets );

    calc_hypercon_args( hycon, box );
    alloc_managed_hypercon( hycon );
    gpu_make_hypercon( hycon, con, radius, box);

    return 0;
    }
