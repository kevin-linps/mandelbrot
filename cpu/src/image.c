#include <stdio.h>
#include <stdlib.h>

#include "image.h"

void image_init(image_t *img, int width, int height)
{
    img->width = width;
    img->height = height;
    img->data = malloc(width * height * sizeof(pixel_t));
}

void image_free(image_t *img)
{
    free(img->data);
}

void image_write(image_t *img, const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (f == NULL)
    {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        return;
    }

    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    for (int i = 0; i < img->width * img->height; i++)
    {
        fwrite(&img->data[i], sizeof(pixel_t), 1, f);
    }

    fclose(f);
}
