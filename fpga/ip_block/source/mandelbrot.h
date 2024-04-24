#ifndef MANDELBROT_H
#define MANDELBROT_H

#include "fixed_point.h"

#define MAX_ITER 128
#define IP_BLOCK_W 128

typedef struct {
	uint8_t b;
	uint8_t g;
	uint8_t r;
} rgb_pixel_t;

void mbrot_compute(fixed real[IP_BLOCK_W], fixed imag, rgb_pixel_t img[IP_BLOCK_W]);

#endif
