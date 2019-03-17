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
cudaError_t gpu_zero_confv( tpvec *thdconfv, tpbox tbox );
cudaError_t gpu_update_vr( tpvec *thdcon, tpvec *thdconv, tpvec *thdconf, tpbox tbox, double dt);
cudaError_t gpu_update_v( tpvec *thdconv, tpvec *thdconf, tpbox tbox, double dt);
cudaError_t gpu_calc_force( tpvec *thdconf, tplist thdlist, tpvec *thdcon, double *thdradius, double *static_press, tpbox tbox );
double gpu_calc_fmax( tpvec *thdconf, tpbox tbox );

#endif
