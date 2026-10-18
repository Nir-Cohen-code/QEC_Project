cmake_minimum_required(VERSION 3.28)
project(QEC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ubuntu/WSL: sudo apt install libntl-dev libgmp-dev
find_path(NTL_INCLUDE_DIR NAMES NTL/mat_GF2.h REQUIRED)
find_library(NTL_LIBRARY NAMES ntl REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)
find_package(Threads REQUIRED)

add_executable(QEC main.cpp
        find_d_monitored.hpp)
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(QEC PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(QEC PRIVATE "${NTL_INCLUDE_DIR}")
target_link_libraries(QEC PRIVATE
        "${NTL_LIBRARY}" "${GMP_LIBRARY}" Threads::Threads)