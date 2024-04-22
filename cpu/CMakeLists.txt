cmake_minimum_required(VERSION 3.10)
project(mandelbrot C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

# Add your source files
file(GLOB SOURCES "src/*.c")
add_executable(mandelbrot ${SOURCES})

# Link the libraries to your executable
target_link_libraries(mandelbrot m)
