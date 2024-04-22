#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <stdint.h>

#include "fixed_point.h"
#include "image.h"

#define MAX_ITER 128

typedef enum 
{
    MBROT_ITER_CMPLX,
    MBROT_ITER_FLOAT,
    MBROT_ITER_FIXED
} mbrot_iter_t;

void mbrot_compute_cmplx(float *real, float *imag, image_t *img);
void mbrot_compute_float(float *real, float *imag, image_t *img);
void mbrot_compute_fixed(fixed *real, fixed *imag, image_t *img);

#endif