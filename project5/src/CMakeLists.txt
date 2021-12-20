cmake_minimum_required(VERSION 3.16)

project(cnn)

include_directories(/opt/OpenBLAS/include)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

link_libraries("/opt/OpenBLAS/lib/libopenblas.so")

add_executable(cnn cnn.cpp)

target_link_libraries(cnn ${OpenCV_LIBS})
