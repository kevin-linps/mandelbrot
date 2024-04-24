#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <stdint.h>
#include "common/mjpeg423_types.h"
#include "fixed_point.h"

#define MAX_ITER 128
#define PLOT_W 1280
#define PLOT_H 720

typedef struct
{
	float x_min;
	float x_max;
	float y_min;
	float y_max;
} plot_t;

void mbrot_compute_fixed_cpu(plot_t *plot, rgb_pixel_t *img);

void mbrot_compute_fixed_fpga(plot_t *plot, rgb_pixel_t *img);

#endif
