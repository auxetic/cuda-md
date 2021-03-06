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
#define SHARE_BLOCK_SIZE 512
// subroutine
void mini_fire_cv( vec_t *tcon, double *tradius, box_t tbox );
void mini_fire_cp( vec_t *tcon, double *tradius, box_t *tbox, double targe_press );

#endif
