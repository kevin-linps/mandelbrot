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
    return iter == MAX_ITER ? black : color_palette[iter % COLOR_PALETTE_SIZE];
}

void mbrot_compute(fixed real[IP_BLOCK_W], fixed imag, rgb_pixel_t img[IP_BLOCK_W])
{
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=real
#pragma HLS INTERFACE axis register both port=img

	fixed real_tmp[IP_BLOCK_W];
	rgb_pixel_t img_tmp[IP_BLOCK_W];

#pragma HLS ARRAY_RESHAPE variable=real     type=cyclic factor=4
#pragma HLS ARRAY_RESHAPE variable=real_tmp type=cyclic factor=4
#pragma HLS ARRAY_RESHAPE variable=img      type=cyclic factor=4
#pragma HLS ARRAY_RESHAPE variable=img_tmp  type=cyclic factor=4

#pragma HLS DATAFLOW

	loop_copy1: for (int i = 0; i < IP_BLOCK_W; i++)
	{
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=4
		real_tmp[i] = real[i];
	}

	loop_main: for (int j = 0; j < IP_BLOCK_W; j++)
	{
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=4
		fixed x, y, x2, y2, cx, cy;
		uint8_t iteration = MAX_ITER;

		x = y = x2 = y2 = 0;
		cx = real_tmp[j];
		cy = imag;
		for (int iter = 0; iter < MAX_ITER; iter++)
		{
			if (iteration == MAX_ITER)
			{
				y = fixed_mul((x + x), y) + cy;
				x = x2 - y2 + cx;
				x2 = fixed_mul(x, x);
				y2 = fixed_mul(y, y);
				if (x2 + y2 > float2fixed(4.0f))
				{
					iteration = iter;
				}
			}
		}

		img_tmp[j] = mbrot_iter_to_pixel(iteration);
	}

	loop_copy2: for (int k = 0; k < IP_BLOCK_W; k++)
	{
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=4
		img[k] = img_tmp[k];
	}
}
