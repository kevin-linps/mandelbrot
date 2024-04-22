#include <complex.h>
#include <limits.h>
#include <math.h>

#include "mandelbrot.h"

#define COLOR_PALETTE_SIZE 16
static const pixel_t black = {0, 0, 0};
static const pixel_t color_palette[COLOR_PALETTE_SIZE] = {
    {9, 1, 47}, {4, 4, 73}, {0, 7, 100}, {12, 44, 138}, {24, 82, 177}, {57, 125, 209}, {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95}, {255, 170, 0}, {204, 128, 0}, {153, 87, 0}, {106, 52, 3}, {66, 30, 15}, {25, 7, 26}};

pixel_t mbrot_iter_to_pixel(uint8_t iter)
{
    return iter == MAX_ITER ? black : color_palette[iter % 16];
}

void mbrot_compute_cmplx(float *real, float *imag, image_t *img)
{
    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {
            uint8_t iteration;
            float complex c = CMPLXF(real[j], imag[i]);
            float complex z = c;
            iteration = MAX_ITER;
            for (iteration = 0; iteration < MAX_ITER; iteration++)
            {
                z = z * z + c;
                if (creal(z) * creal(z) + cimag(z) * cimag(z) > 4.0f)
                {
                    break;
                }
            }

            int index = i * img->width + j;
            img->data[index] = mbrot_iter_to_pixel(iteration);
        }
    }
}

void mbrot_compute_float(float *real, float *imag, image_t *img)
{
    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
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

            int index = i * img->width + j;
            img->data[index] = mbrot_iter_to_pixel(iteration);
        }
    }
}

void mbrot_compute_fixed(fixed *real, fixed *imag, image_t *img)
{
    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
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

            int index = i * img->width + j;
            img->data[index] = mbrot_iter_to_pixel(iteration);
        }
    }
}
