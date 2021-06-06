cmake_minimum_required(VERSION 2.8)
project(EigenDemo)

# Check for c++11 support
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  MESSAGE(ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE Debug)

find_package (Eigen3 REQUIRED)

MESSAGE("EIGEN3_FOUND: " ${EIGEN3_FOUND})
MESSAGE("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
MESSAGE("EIGEN3_VERSION: " ${EIGEN3_VERSION})
include_directories(${EIGEN3_INCLUDE_DIR})

MESSAGE("orocos_kdl_INCLUDE_DIRS: " ${orocos_kdl_INCLUDE_DIRS})
MESSAGE("orocos_kdl_LIBRARIES: " ${orocos_kdl_LIBRARIES})
include_directories(${orocos_kdl_INCLUDE_DIRS})

find_package(orocos_kdl)
IF(${orocos_kdl_FOUND})
    MESSAGE("orocos_kdl_LIBRARIES: " ${orocos_kdl_LIBRARIES})
    MESSAGE("orocos_kdl_INCLUDE_DIRS: " ${orocos_kdl_INCLUDE_DIRS})
    MESSAGE("orocos_kdl_LIBRARIES: " ${orocos_kdl_LIBRARIES})
    include_directories(${orocos_kdl_INCLUDE_DIRS})
ENDIF()

add_executable(main src/main.cpp)
target_link_libraries (main ${EIGEN3_LIBRARIES}  ${orocos_kdl_LIBRARIES})
target_compile_definitions(main PRIVATE KDL_FOUND=${orocos_kdl_FOUND})


add_executable(task src/task.cpp)
target_link_libraries (task ${EIGEN3_LIBRARIES} )


#IF(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/sandbox")
#	ADD_SUBDIRECTORY(src/sandbox)
#ENDIF()
