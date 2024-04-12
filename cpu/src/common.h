#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

#define SCREEN_W 1920
#define SCREEN_H 1080

typedef int32_t fixed;
#define FIXED_SHIFT 24
#define FIXED_ONE (1 << FIXED_SHIFT)
#define float2fixed(f) ((fixed)((f) * FIXED_ONE))
#define fixed2float(f) ((float)(f) / FIXED_ONE)
#define fixed_mul(a, b) (((a) >> (FIXED_SHIFT / 2)) * ((b) >> (FIXED_SHIFT / 2)))

typedef struct
{
    uint8_t r, g, b;
} pixel_t;

#endif