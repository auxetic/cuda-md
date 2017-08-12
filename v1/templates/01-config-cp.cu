#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "system.h"
#include "config.h"
#include "list.h"
#include "fire.h"

int main(void)
    {

    int deviceno, natom, seed;
    double press;
    char foutput[100];

    scanf("%d", &deviceno);
    scanf("%d", &natom );
    scanf("%d", &seed );
    scanf("%le", &press );
    scanf("%s", foutput );

    cudaSetDevice(deviceno);
    // set
    box.natom = natom;
    box.phi = 0.86;
    sets.seed = seed;

    printf("%d\n", deviceno);
    printf("%d\n", natom);
    printf("%d\n", seed);
    printf("%e\n", press);
    printf("%s\n", foutput);

//exit(0);
    // cpu config
    //printf( "allocate *con, generate a random config\n" );
    alloc_con( &con, &radius, box.natom );
    gen_config( con, radius, &box, sets );

    // fire
    mini_fire_cp( con, radius, &box, press );

    FILE *fio;
    fio = fopen(foutput, "w+");
        fprintf( fio, "%d %26.16e \n", box.natom, box.x );
        for ( int i=0; i<box.natom; i++ )
            fprintf( fio, "%26.16e  %26.16e  %26.16e \n", con[i].x*box.xinv, con[i].y*box.yinv, radius[i]*box.xinv );
    fclose(fio);

    return 0;
    }
    //// fire on gpu
    //mini_fire_cv( con, radius, box );
