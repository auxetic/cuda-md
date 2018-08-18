#ifndef md_h
#define md_h

#include "system.h"
#include "config.h"
#include "list.h"
#include "mdfunc.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudamath.h"

#include <stdio.h>
#include <memory.h>
#include <stdbool.h>

// type define
typedef struct
    {
    double temper;
    double press;
    } tpmdset;

// variable define
extern tpmdset mdset;

void init_nvt( vec_t *thcon, double *thradius, box_t tbox, double ttemper );
void gpu_run_nvt( vec_t *tdcon, vec_t *tdconv, vec_t *tdconf, double ttemper, int steps );


#endif
