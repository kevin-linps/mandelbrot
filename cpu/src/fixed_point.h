#ifndef FIXED_POINT_H
#define FIXED_POINT_H

#include <stdint.h>

typedef int32_t fixed;
#define FIXED_SHIFT 24
#define FIXED_ONE (1 << FIXED_SHIFT)
#define float2fixed(f) ((fixed)((f) * FIXED_ONE))
#define fixed2float(f) ((float)(f) / FIXED_ONE)
#define fixed_mul(a, b) (((a) >> (FIXED_SHIFT / 2)) * ((b) >> (FIXED_SHIFT / 2)))

#endif