#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "image.h"
#include "mandelbrot.h"

static mbrot_iter_t iter_type = MBROT_ITER_CMPLX;
static float x_min = -2.52f, x_max = 1.0f, y_min = -0.99f, y_max = 0.99f;
static int img_w = 1920, img_h = 1080;

void parse_args(int argc, char *argv[]);
void generate_array_float(float *arr, int size, float start, float end);
void generate_array_fixed(fixed *arr, int size, float start, float end);

int main(int argc, char *argv[])
{
    parse_args(argc, argv);

    image_t pic;
    image_init(&pic, img_w, img_h);

    float *real, *imag;
    real = malloc(img_w * sizeof(float));
    imag = malloc(img_h * sizeof(float));

    clock_t start = clock();
    switch (iter_type)
    {
    case MBROT_ITER_CMPLX:
        generate_array_float(real, img_w, x_min, x_max);
        generate_array_float(imag, img_h, y_min, y_max);
        mbrot_compute_cmplx(real, imag, &pic);
        break;
    case MBROT_ITER_FLOAT:
        generate_array_float(real, img_w, x_min, x_max);
        generate_array_float(imag, img_h, y_min, y_max);
        mbrot_compute_float(real, imag, &pic);
        break;
    case MBROT_ITER_FIXED:
        generate_array_fixed((fixed *)real, img_w, x_min, x_max);
        generate_array_fixed((fixed *)imag, img_h, y_min, y_max);
        mbrot_compute_fixed((fixed *)real, (fixed *)imag, &pic);
        break;
    default:
        printf("Invalid iteration method\n");
        return -1;
    }
    clock_t end = clock();
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    image_write(&pic, "mandelbrot.ppm");
    image_free(&pic);

    free(real);
    free(imag);

    return 0;
}

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "c:w:h:")) != -1)
    {
        switch (opt)
        {
        case 'c':
            if (strcmp(optarg, "complex") == 0)
            {
                iter_type = MBROT_ITER_CMPLX;
            }
            else if (strcmp(optarg, "float") == 0)
            {
                iter_type = MBROT_ITER_FLOAT;
            }
            else if (strcmp(optarg, "fixed") == 0)
            {
                iter_type = MBROT_ITER_FIXED;
            }
            break;
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

void generate_array_float(float *arr, int size, float start, float end)
{
    float delta = (end - start) / size;
    for (int i = 0; i < size; i++)
    {
        arr[i] = start + i * delta;
    }
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
