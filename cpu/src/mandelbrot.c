#include <complex.h>
#include <limits.h>
#include <math.h>

#include "mandelbrot.h"

void mbrot_iter_cmplx(float real[SCREEN_W], float imag[SCREEN_H], uint8_t iter[SCREEN_H][SCREEN_W])
{
    float complex c, z;
    float magn;
    uint8_t iteration;
    for (int i = 0; i < SCREEN_H; i++)
    {
        for (int j = 0; j < SCREEN_W; j++)
        {
            c = CMPLXF(real[j], imag[i]);
            z = c;
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
    float x, y, x2, y2, cx, cy;
    uint8_t iteration;
    for (int i = 0; i < SCREEN_H; i++)
    {
        for (int j = 0; j < SCREEN_W; j++)
        {
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
}

void mbrot_iter_to_pixel(uint8_t iter[SCREEN_H][SCREEN_W], uint8_t pic[SCREEN_H][SCREEN_W][RGB])
{
    float ratio;
    for (int i = 0; i < SCREEN_H; i++)
    {
        for (int j = 0; j < SCREEN_W; j++)
        {
            ratio = (float)iter[i][j] / MAX_ITER;
            pic[i][j][R] = 9.00f * powf(1.0f - ratio, 1) * powf(ratio, 3) * UINT8_MAX;
            pic[i][j][G] = 15.0f * powf(1.0f - ratio, 2) * powf(ratio, 2) * UINT8_MAX;
            pic[i][j][B] = 8.50f * powf(1.0f - ratio, 3) * powf(ratio, 1) * UINT8_MAX;
            
            // pic[i][j][R] = iter[i][j] == MAX_ITER ? UINT8_MAX : 0;
            // pic[i][j][G] = iter[i][j] == MAX_ITER ? UINT8_MAX : 0;
            // pic[i][j][B] = iter[i][j] == MAX_ITER ? UINT8_MAX : 0;
        }
    }
}
