cmake_minimum_required(VERSION 3.10)
project(mandelbrot LANGUAGES CXX CUDA)

set(CMAKE_C_STANDARD 11)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/x86_64-linux-gnu")

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_60")

include_directories(${CUDA_INCLUDE_DIRS})

# Add your source files here
file(GLOB SOURCES "src/*.cu")

CUDA_ADD_EXECUTABLE(mandelbrot ${SOURCES})

target_link_libraries(mandelbrot ${CUDA_LIBRARIES})