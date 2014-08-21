cmake_minimum_required(VERSION 2.8)
project(EigenDemo)



set(CMAKE_BUILD_TYPE Debug)

find_package (Eigen3 REQUIRED)
find_package (orocos_kdl REQUIRED)

MESSAGE("EIGEN3_FOUND: " ${EIGEN3_FOUND})
MESSAGE("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
MESSAGE("EIGEN3_VERSION: " ${EIGEN3_VERSION})
include_directories(${EIGEN3_INCLUDE_DIR})

MESSAGE("orocos_kdl_LIBRARIES: " ${orocos_kdl_LIBRARIES})


add_executable(main src/main.cpp)
target_link_libraries (main ${EIGEN3_LIBRARIES}  ${orocos_kdl_LIBRARIES})



