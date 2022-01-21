cmake_minimum_required(VERSION 3.22)

project(wordle LANGUAGES CUDA)

add_executable(wordle src/main.cu)
target_compile_features(wordle PRIVATE cuda_std_20)
