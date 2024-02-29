cmake_minimum_required(VERSION 3.10)
project(mandelbrot C)

set(CMAKE_C_STANDARD 11)

# Add your source files
file(GLOB SOURCES "src/*.c")
add_executable(mandelbrot ${SOURCES})

# Add any libraries in the lib folder
file(GLOB LIBRARIES "lib/*.c")
add_library(bmp ${LIBRARIES})

# Link the libraries to your executable
target_link_libraries(mandelbrot bmp m)
