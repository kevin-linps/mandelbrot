# Mandelbrot Set

Mandelbrot set is a mathematical fractal on complex plane. Rendering it requires lots of computation, but it is an embarassingly parallel task. This repository contains multiple implementations of mandelbrot set rendering algorithms on different devices like CPU, GPU and FPGA.


## CPU Implementation

### Build Instruction

```
cd cpu
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Benchmark

There are 3 CPU implementations. Each can be invoked with `./mandelbrot -c ${mode} -w {image_width} -h {image_height}`.

1. `complex`: Uses `complex.h` from standard C library to perform complex number calculations.
2. `float`: Uses float with optimized instructions to perform calculation (see [Wikipedia](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Optimized_escape_time_algorithms))
3. `fixed`: Same implementation as `float` but uses fixed-point integer with radix at 24th bit instead.


The table below is obtained using [hyperfine](https://github.com/sharkdp/hyperfine) wuth 3 warm-up runs and 10 runs. The unit is milliseconds, so the lower the better.

| Implementation | FHD (1920 x 1080) | 2K (2560 x 1440) | 4K (3840 x 2160) |
| -------------- | ----------------- | ---------------- | ---------------- |
| `complex`      | 390.0             | 681.1            | 1545             |
| `float`        | 322.7             | 576.1            | 1289             |
| `fixed`        | 242.1             | 428.7            | 956.6            |
