#include <stdio.h>
#include <device_launch_parameters.h>

#include "mandelbrot.cuh"

#define COLOR_PALETTE_SIZE 16
__constant__ uchar3 black = {0, 0, 0};
__constant__ uchar3 color_palette[COLOR_PALETTE_SIZE] = {
    {9, 1, 47}, {4, 4, 73}, {0, 7, 100}, {12, 44, 138}, {24, 82, 177}, {57, 125, 209}, {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95}, {255, 170, 0}, {204, 128, 0}, {153, 87, 0}, {106, 52, 3}, {66, 30, 15}, {25, 7, 26}};

__device__ uchar3 mbrot_iter_to_pixel(int iter)
{
    return iter == MAX_ITER ? black : color_palette[iter % 16];
}

__global__ void mbrot_compute_float(float4 plot_area, int2 img_size, uchar3 *pixels)
{
    float x_min = plot_area.x;
    float x_max = plot_area.y;
    float y_min = plot_area.z;
    float y_max = plot_area.w;
    int img_w = img_size.x;
    int img_h = img_size.y;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float cx = x_min + (i % img_w) * (x_max - x_min) / img_w;
    float cy = y_min + (i / img_w) * (y_max - y_min) / img_h;

    float x = 0.0f, y = 0.0f, x2 = 0.0f, y2 = 0.0f;
    int iteration;

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

    pixels[i] = mbrot_iter_to_pixel(iteration);
}
