#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <stdint.h>

#include "common.h"

#define MAX_ITER 128

typedef enum 
{
    MBROT_ITER_CMPLX,
    MBROT_ITER_FLOAT,
    MBROT_ITER_FIXED
} mbrot_iter_t;

void mbrot_iter_cmplx(float real[SCREEN_W], float imag[SCREEN_H], uint8_t iter[SCREEN_H][SCREEN_W]);
void mbrot_iter_float(float real[SCREEN_W], float imag[SCREEN_H], uint8_t iter[SCREEN_H][SCREEN_W]);
void mbrot_iter_fixed(fixed real[SCREEN_W], fixed imag[SCREEN_H], uint8_t iter[SCREEN_H][SCREEN_W]);

void mbrot_iter_to_pixel(uint8_t iter[SCREEN_H][SCREEN_W], pixel_t pic[SCREEN_H][SCREEN_W]);

#endif