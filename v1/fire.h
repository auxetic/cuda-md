#ifndef fire_h
#define fire_h

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

// const
#define SHARE_BLOCK_SIZE 1024
// subroutine
void mini_fire_cv( tpvec *tcon, double *tradius, tpbox tbox );
cudaError_t gpu_calc_force( tplist *tdlist, tpvec *tdcon, double *tddradius, tpvec *tdconf, tpbox tbox );
cudaError_t gpu_calc_fire_para( tpvec *tdconv, tpvec *tdconf, tpbox tbox );
cudaError_t gpu_fire_modify_v( tpvec *tdconv, tpvec *tdconf, double tfire_onemb, double tfire_betamvndfn, tpbox tbox);

#endif
