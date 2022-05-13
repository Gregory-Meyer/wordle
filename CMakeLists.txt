cmake_minimum_required(VERSION 3.17)

project(wordle LANGUAGES CUDA)

add_executable(wordle
    src/main.cu

    src/io.cu
    src/result.cu
    src/util.cu
    src/word.cu
)
target_compile_features(wordle PRIVATE cuda_std_17)
