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
cudaError_t gpu_zero_confv( vec_t *thdconfv, box_t tbox );
cudaError_t gpu_update_vr( vec_t *thdcon, vec_t *thdconv, vec_t *thdconf, box_t tbox, double dt);
cudaError_t gpu_update_v( vec_t *thdconv, vec_t *thdconf, box_t tbox, double dt);
cudaError_t gpu_calc_force( vec_t *thdconf, list_t thdlist, vec_t *thdcon, double *thdradius, double *static_press, box_t tbox );
double gpu_calc_fmax( vec_t *thdconf, box_t tbox );

#endif
