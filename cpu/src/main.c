#include <stdio.h>
#include <time.h>

#include "../lib/libbmp.h"

#include "common.h"
#include "mandelbrot.h"

int main(int argc, char *argv[])
{
    float x[SCREEN_W], y[SCREEN_H];
    uint8_t iter[SCREEN_H][SCREEN_W];
    uint8_t pic[SCREEN_H][SCREEN_W][RGB];

    for (int i = 0; i < SCREEN_W; i++)
    {
        x[i] = -2.0f + i * (3.0f / SCREEN_W);
    }
    for (int i = 0; i < SCREEN_H; i++)
    {
        y[i] = -1.0f + i * (2.0f / SCREEN_H);
    }
    clock_t start = clock();
    mbrot_iter_float(x, y, iter);
    clock_t end = clock();
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    mbrot_iter_to_pixel(iter, pic);

    bmp_img img;
    bmp_img_init_df(&img, SCREEN_W, SCREEN_H);
    for (size_t x = 0; x < SCREEN_H; x++)
    {
        for (size_t y = 0; y < SCREEN_W; y++)
        {
            bmp_pixel_init(&img.img_pixels[x][y], pic[x][y][R], pic[x][y][G], pic[x][y][B]);
        }
    }
    bmp_img_write(&img, "test.bmp");
    bmp_img_free(&img);

    return 0;
}
