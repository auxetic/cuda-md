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
void mini_fire_cv( tpvec *tcon, double *tradius, tpbox tbox );
void mini_fire_cp( tpvec *tcon, double *tradius, tpbox *tbox, double targe_press );

#endif
