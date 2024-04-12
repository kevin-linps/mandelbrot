#include <complex.h>
#include <limits.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mandelbrot.h"

/* Colour palette
 * ratio = iteration / MAX_ITER
 * r = 9.00 * pow(1.0 - ratio, 1) * pow(ratio, 3) * 255
 * g = 15.0 * pow(1.0 - ratio, 2) * pow(ratio, 2) * 255
 * b = 8.50 * pow(1.0 - ratio, 3) * pow(ratio, 1) * 255
 */
static uint8_t r[] = {9, 4, 0, 12, 24, 57, 134, 211, 241, 248, 255, 204, 153, 106, 66, 25};
static uint8_t g[] = {1, 4, 7, 44, 82, 125, 181, 236, 233, 201, 170, 128, 87, 52, 30, 7};
static uint8_t b[] = {47, 73, 100, 138, 177, 209, 229, 248, 191, 95, 0, 0, 0, 3, 15, 26};

void mbrot_iter_cmplx(float real[SCREEN_W], float imag[SCREEN_H], uint8_t iter[SCREEN_H][SCREEN_W])
{
#pragma omp parallel for collapse(2) num_threads(16)
    for (int i = 0; i < SCREEN_H; i++)
    {
        for (int j = 0; j < SCREEN_W; j++)
        {
            uint8_t iteration;
            float complex c = CMPLXF(real[j], imag[i]);
            float complex z = c;
            iteration = MAX_ITER;
            for (iteration = 0; iteration < MAX_ITER; iteration++)
            {
                z = z * z + c;
                if (cabsf(z) > 2)
                {
                    break;
                }
            }
            iter[i][j] = iteration;
        }
    }
}

void mbrot_iter_float(float real[SCREEN_W], float imag[SCREEN_H], uint8_t iter[SCREEN_H][SCREEN_W])
{
#pragma omp parallel for collapse(2) num_threads(16)
    for (int i = 0; i < SCREEN_H; i++)
    {
        for (int j = 0; j < SCREEN_W; j++)
        {
            float x, y, x2, y2, cx, cy;
            uint8_t iteration;

            x = y = x2 = y2 = 0.0f;
            cx = real[j];
            cy = imag[i];
            iteration = MAX_ITER;
            for (iteration = 0; iteration < MAX_ITER; iteration++)
            {
                y = (x + x) * y + cy;
                x = x2 - y2 + cx;
                x2 = x * x;
                y2 = y * y;
                if (x2 + y2 > 4)
                {
                    break;
                }
            }
            iter[i][j] = iteration;
        }
    }
}

void mbrot_iter_fixed(fixed real[SCREEN_W], fixed imag[SCREEN_H], uint8_t iter[SCREEN_H][SCREEN_W])
{
#pragma omp parallel for collapse(2) num_threads(16)
    for (int i = 0; i < SCREEN_H; i++)
    {
        for (int j = 0; j < SCREEN_W; j++)
        {
            fixed x, y, x2, y2, cx, cy;
            uint8_t iteration;

            x = y = x2 = y2 = 0;
            cx = real[j];
            cy = imag[i];
            iteration = MAX_ITER;
            for (iteration = 0; iteration < MAX_ITER; iteration++)
            {
                y = fixed_mul((x + x), y) + cy;
                x = x2 - y2 + cx;
                x2 = fixed_mul(x, x);
                y2 = fixed_mul(y, y);
                if (x2 + y2 > float2fixed(4.0f))
                {
                    break;
                }
            }
            iter[i][j] = iteration;
        }
    }
}

void mbrot_iter_to_pixel(uint8_t iter[SCREEN_H][SCREEN_W], pixel_t pic[SCREEN_H][SCREEN_W])
{
#pragma omp parallel for collapse(2) num_threads(16)
    for (int i = 0; i < SCREEN_H; i++)
    {
        for (int j = 0; j < SCREEN_W; j++)
        {
            if (iter[i][j] == MAX_ITER)
            {
                pic[i][j].r = 0;
                pic[i][j].g = 0;
                pic[i][j].b = 0;
            } else {
                // int idx = iter[i][j] * 16 / MAX_ITER;
                int idx = iter[i][j] % 16;
                pic[i][j].r = r[idx];
                pic[i][j].g = g[idx];
                pic[i][j].b = b[idx];
            }
        }
    }
}
