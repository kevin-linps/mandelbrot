#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <cuda_runtime.h>
#include "image.cuh"

#define MAX_ITER 128

__global__ void mbrot_compute_float(float4 plot_area, int2 img_size, uchar3 *pixels);

#endif