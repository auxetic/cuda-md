CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cumd)

FIND_PACKAGE(CUDA REQUIRED)

FILE(GLOB SOURCES "*.c" "*.cc" "*.cu")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
#Debug
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -O0 -std=c++11 -gencode arch=compute_60,code=sm_60 --ptxas-options --verbose) 
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 -Xptxas -O3 -Wno-deprecated-gpu-targets ) 
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_60 ) 
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_60 -Wno-deprecated-gpu-targets )
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -rdc=true )
set(CMAKE_C_FLAGS "-std=c99") 

CUDA_ADD_EXECUTABLE(main ${SOURCES})
