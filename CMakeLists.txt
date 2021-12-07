cmake_minimum_required(VERSION 3.20)
project(Lab5_Cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Lab5_Cuda main.cu)

set_target_properties(Lab5_Cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
