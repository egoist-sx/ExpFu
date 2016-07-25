cmake_minimum_required(VERSION 2.8)
project(ExpFu)
find_package(OpenCV REQUIRED)

add_executable(ExpFu ${PROJECT_SOURCE_DIR}/src/main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ExpFu ${OpenCV_LIBS})
