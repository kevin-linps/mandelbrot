#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

typedef struct
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
} pixel_t;

typedef struct
{
    int width;
    int height;
    pixel_t *data;
} image_t;

void image_init(image_t *img, int width, int height);
void image_free(image_t *img);

// Write image to a PPM file
void image_write(image_t *img, const char *filename);

#endif