#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <getopt.h>

#include <cuda.h>

#include "image.cuh"
#include "mandelbrot.cuh"

static float x_min = -2.52f, x_max = 1.0f, y_min = -0.99f, y_max = 0.99f;
static int img_w = 1920, img_h = 1080;

#define BLOCK_SIZE 256

void parse_args(int argc, char *argv[]);
void generate_array_float(float *arr, int size, float start, float end);

int main(int argc, char *argv[])
{
    parse_args(argc, argv);

    image_t pic;
    image_init(&pic, img_w, img_h);

    clock_t start = clock();
    float4 plot_area = make_float4(x_min, x_max, y_min, y_max);
    int2 img_size = make_int2(img_w, img_h);
    uchar3 *pixel;
    cudaMalloc((void **) &pixel, img_w * img_h * sizeof(uchar3));
    mbrot_compute_float<<<img_w * img_h / BLOCK_SIZE, BLOCK_SIZE>>>(plot_area, img_size, pixel);
    cudaDeviceSynchronize();
    cudaMemcpy(pic.data, pixel, img_w * img_h * sizeof(uchar3), cudaMemcpyDeviceToHost);
    clock_t end = clock();
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    image_write(&pic, "mandelbrot.ppm");
    image_free(&pic);

    cudaFree(pixel);

    return 0;
}

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "w:h:")) != -1)
    {
        switch (opt)
        {
        case 'w':
            img_w = atoi(optarg);
            break;
        case 'h':
            img_h = atoi(optarg);
            break;
        default:
            break;
        }
    }
}
