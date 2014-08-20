cmake_minimum_required(VERSION 2.8)
project(EigenDemo)


set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/behnam/workspace/EigenBuild/usr/local")
MESSAGE( "CMAKE_PREFIX_PATH : " ${CMAKE_PREFIX_PATH} )

set(CMAKE_BUILD_TYPE Debug)

#This Line will add FindEigen3.cmake to our cmake file, because it doesn't exist in /usr/share/cmake-2.8/Modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package (Eigen3 REQUIRED)
add_executable(main src/main.cpp)	
if (EIGEN3_FOUND)
  MESSAGE("EIGEN3_FOUND: " ${EIGEN3_FOUND})
  MESSAGE("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})	
  MESSAGE("EIGEN3_VERSION: " ${EIGEN3_VERSION})
 include_directories(${EIGEN3_INCLUDE_DIR})
#  include_directories("/usr/include/eigen3")
  target_link_libraries (main ${EIGEN3_LIBRARIES} )
endif (EIGEN3_FOUND)


