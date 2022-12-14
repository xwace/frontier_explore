cmake_minimum_required(VERSION 3.23)
project(frontier_explore)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable(frontier_explore main.cpp)
target_link_libraries(frontier_explore ${OpenCV_LIBS})
