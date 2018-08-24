cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tensorrt-cmake LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDA REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${GIE_PATH}/include)
include_directories(utils)

file(GLOB alexnetrtsources src/alexnetRT.cpp)
add_executable(alexnetrt ${alexnetrtsources})
target_link_libraries(alexnetrt nvcaffe_parser nvinfer ${CUDA_LIBRARIES})