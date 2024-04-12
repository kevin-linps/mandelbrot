#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "../lib/libbmp.h"

#include "mandelbrot.h"

int main(int argc, char *argv[])
{
    float x[SCREEN_W], y[SCREEN_H];
    fixed x_fixed[SCREEN_W], y_fixed[SCREEN_H];
    uint8_t iter[SCREEN_H][SCREEN_W];
    pixel_t pic[SCREEN_H][SCREEN_W];
    mbrot_iter_t iter_type = MBROT_ITER_FLOAT;

    int opt;
    while ((opt = getopt(argc, argv, "cfx")) != -1)
    {
        switch (opt)
        {
        case 'c':
            iter_type = MBROT_ITER_CMPLX;
            break;
        case 'f':
            iter_type = MBROT_ITER_FLOAT;
            break;
        case 'x':
            iter_type = MBROT_ITER_FIXED;
            break;
        default:
            break;
        }
    }

    clock_t start = clock();
    switch (iter_type)
    {
    case MBROT_ITER_CMPLX:
        /* code */
        for (int i = 0; i < SCREEN_W; i++)
            x[i] = -2.5f + i * (3.556f / SCREEN_W);
        for (int i = 0; i < SCREEN_H; i++)
            y[i] = -1.0f + i * (2.0f / SCREEN_H);
        mbrot_iter_cmplx(x, y, iter);
        break;
    case MBROT_ITER_FLOAT:
        /* code */
        for (int i = 0; i < SCREEN_W; i++)
            x[i] = -2.5f + i * (3.556f / SCREEN_W);
        for (int i = 0; i < SCREEN_H; i++)
            y[i] = -1.0f + i * (2.0f / SCREEN_H);
        mbrot_iter_float(x, y, iter);
        break;
    case MBROT_ITER_FIXED:
        /* code */
        for (int i = 0; i < SCREEN_W; i++)
            x_fixed[i] = float2fixed(-2.5f + i * (3.556f / SCREEN_W));
        for (int i = 0; i < SCREEN_H; i++)
            y_fixed[i] = float2fixed(-1.0f + i * (2.000f / SCREEN_H));
        mbrot_iter_fixed(x_fixed, y_fixed, iter);
        break;
    default:
        break;
    }
    clock_t end = clock();
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    mbrot_iter_to_pixel(iter, pic);

    bmp_img img;
    bmp_img_init_df(&img, SCREEN_W, SCREEN_H);
    for (size_t x = 0; x < SCREEN_H; x++)
    {
        for (size_t y = 0; y < SCREEN_W; y++)
        {
            bmp_pixel_init(&img.img_pixels[x][y], pic[x][y].r, pic[x][y].g, pic[x][y].b);
        }
    }
    bmp_img_write(&img, "mandelbrot.bmp");
    bmp_img_free(&img);

    return 0;
}
