/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <inttypes.h>
#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xtime_l.h"

#include "common/util.h"
#include "ece423_vid_ctl/ece423_vid_ctl.h"
#include "mandelbrot.h"


int main()
{
    init_platform();

    print("Welcome to Mandelbrot Set Visualizer!\n");

    XTime t1, t2;
    XTime_GetTime(&t1);

    vdma_init(PLOT_W, PLOT_H, 5);

    rgb_pixel_t* rgbblock;
    rgbblock = buff_next();

    plot_t plot = { .x_min = -2.52f, .x_max = 1.00f, .y_min = -0.99f, .y_max = 0.99f };
    mbrot_compute_fixed_cpu(&plot, rgbblock);

    buff_reg();
    vdma_out();

    XTime_GetTime(&t2);
    printf("Execution time: %" PRIu64 "ms\n", (t2-t1)*1000/COUNTS_PER_SECOND);

    while(1);

    vdma_close();
    cleanup_platform();
    return 0;
}
