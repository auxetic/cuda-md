#ifndef mdfunc_h
#define mdfunc_h

#include "system.h"
#include "config.h"
#include "list.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudamath.h"

#include <stdio.h>
#include <memory.h>
#include <stdbool.h>

// funcs
cudaError_t gpu_zero_confv( tpvec *tdconfv, tpbox tbox );
cudaError_t gpu_update_vr( tpvec *tdcon, tpvec *tdconv, tpvec *tdconf, tpbox tbox, double dt);
cudaError_t gpu_update_v( tpvec *tdconv, tpvec *tdconf, tpbox tbox, double dt);
cudaError_t gpu_calc_force( tplist *tdlist, tpvec *tdcon, double *tddradius, tpvec *tdconf, double *static_press, tpbox tbox );
double gpu_calc_fmax( tpvec *tdconf, tpbox tbox );

#endif
