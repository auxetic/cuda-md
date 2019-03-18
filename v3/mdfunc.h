#ifndef __mdfunc_h
#define __mdfunc_h

#include "system.h"
#include "config.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudamath.h"

#include <stdio.h>
#include <memory.h>
#include <stdbool.h>

#define BLOCK_SIZE_256  256
#define BLOCK_SIZE_512  512
#define BLOCK_SIZE_1024 1024


cudaError_t gpu_zero_confv( vec_t *thdconfv, box_t tbox );
cudaError_t gpu_update_vr( vec_t *thdcon, vec_t *thdconv, vec_t *thdconf, box_t tbox, double dt);
cudaError_t gpu_update_v( vec_t *thdconv, vec_t *thdconf, box_t tbox, double dt);

cudaError_t gpu_calc_force( vec_t  *thdconf, hycon_t *thdblocks, double  *static_press, box_t tbox );
double gpu_calc_fmax( vec_t *thdconf, box_t tbox );

#endif
