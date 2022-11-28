cmake_minimum_required(VERSION 3.15.0...3.21.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(testTriArborX VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(ArborX REQUIRED)
find_package(Omega_h REQUIRED VERSION 10)

add_executable(testTri testTri.cpp)
target_link_libraries(testTri PRIVATE ArborX::ArborX Omega_h::omega_h)
