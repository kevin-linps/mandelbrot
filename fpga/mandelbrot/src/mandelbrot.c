#include <complex.h>
#include <limits.h>
#include <math.h>

#include "mandelbrot.h"

#define COLOR_PALETTE_SIZE 16
static const rgb_pixel_t black = {0, 0, 0};
static const rgb_pixel_t color_palette[COLOR_PALETTE_SIZE] = { { 9, 1, 47 }, {
		4, 4, 73 }, { 0, 7, 100 }, { 12, 44, 138 }, { 24, 82, 177 }, { 57, 125,
		209 }, { 134, 181, 229 }, { 211, 236, 248 }, { 241, 233, 191 }, { 248,
		201, 95 }, { 255, 170, 0 }, { 204, 128, 0 }, { 153, 87, 0 }, { 106, 52,
		3 }, { 66, 30, 15 }, { 25, 7, 26 } };

rgb_pixel_t mbrot_iter_to_pixel(uint8_t iter)
{
    return iter == MAX_ITER ? black : color_palette[iter % 16];
}

void generate_array_fixed(fixed *arr, int size, float start, float end)
{
    fixed start_fixed = float2fixed(start);
    fixed delta = float2fixed((end - start) / size);
    for (int i = 0; i < size; i++)
    {
        arr[i] = start_fixed + i * delta;
    }
}

void mbrot_compute_fixed_cpu(plot_t *plot, rgb_pixel_t *img)
{
	fixed real[PLOT_W], imag[PLOT_H];
	generate_array_fixed(real, PLOT_W, plot->x_min, plot->x_max);
	generate_array_fixed(imag, PLOT_H, plot->y_min, plot->y_max);

    for (int i = 0; i < PLOT_H; i++)
    {
        for (int j = 0; j < PLOT_W; j++)
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

            int index = i * PLOT_W + j;
            img[index] = mbrot_iter_to_pixel(iteration);
        }
    }
}

void mbrot_compute_fixed_fpga(plot_t *plot, rgb_pixel_t *img)
{

}
