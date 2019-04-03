cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project(projections)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(ipm_projection ipm_projection.cpp)
target_link_libraries(ipm_projection PRIVATE ${OpenCV_LIBRARIES} ::Eigen3)


add_executable(calib_converter calib_converter.cpp)
target_link_libraries(calib_converter PRIVATE ${OpenCV_LIBRARIES} ::Eigen3)

