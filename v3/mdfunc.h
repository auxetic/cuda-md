#ifndef __mdfunc_h
#define __mdfunc_h

#include <stdio.h>
#include <memory.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudamath.h"

#include "system.h"
#include "config.h"

cudaError_t gpu_zero_confv( vec_t *confv, box_t box );
cudaError_t gpu_update_vr( vec_t *con, vec_t *conv, vec_t *conf, box_t tbox, double dt);
cudaError_t gpu_update_v( vec_t *conv, vec_t *conf, box_t box, double dt);

cudaError_t gpu_calc_force( hycon_t *blocks, double *static_press, box_t box );
double gpu_calc_fmax( vec_t *conf, box_t tbox );

#endif
