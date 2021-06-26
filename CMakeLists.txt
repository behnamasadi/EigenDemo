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



#set(Eigen3_DIR "$ENV{HOME}/usr/share/eigen3/cmake")

find_package (Eigen3 REQUIRED NO_MODULE)

MESSAGE("EIGEN3_FOUND: " ${EIGEN3_FOUND})
MESSAGE("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
MESSAGE("EIGEN3_VERSION: " ${EIGEN3_VERSION})
MESSAGE("EIGEN3_VERSION_STRING: " ${EIGEN3_VERSION_STRING})

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

#add_executable(main src/main.cpp)
#target_link_libraries (main)

add_executable (main src/main.cpp)
target_link_libraries (main Eigen3::Eigen)

add_executable(non_linear_regression src/non_linear_regression.cpp)
target_link_libraries (non_linear_regression)

add_executable(non_linear_least_squares src/non_linear_least_squares.cpp)
target_link_libraries (non_linear_least_squares)

add_executable(numerical_diff src/numerical_diff.cpp)
target_link_libraries (numerical_diff)

add_executable(levenberg_marquardt src/levenberg_marquardt.cpp)
target_link_libraries (levenberg_marquardt)

add_executable(eigen_functor src/eigen_functor.cpp)
target_link_libraries (eigen_functor)

add_executable(auto_diff src/auto_diff.cpp)
target_link_libraries (auto_diff)


add_executable(matrix_array_vector src/matrix_array_vector.cpp)
target_link_libraries (matrix_array_vector)


add_executable(unaryExpr src/unaryExpr.cpp)
target_link_libraries (unaryExpr)

add_executable(memory_mapping src/memory_mapping.cpp)
target_link_libraries (memory_mapping)

add_executable(matrix_broadcasting src/matrix_broadcasting.cpp)
target_link_libraries (matrix_broadcasting)

add_executable(matrix_decomposition src/matrix_decomposition.cpp)
target_link_libraries (matrix_decomposition)

add_executable(gram_schmidt_orthogonalization src/gram_schmidt_orthogonalization.cpp)
target_link_libraries (gram_schmidt_orthogonalization)


add_executable(check_matrixsimilarity src/check_matrixsimilarity.cpp)
target_link_libraries (check_matrixsimilarity)

add_executable(geometry_transformation src/geometry_transformation.cpp)
target_link_libraries (geometry_transformation   ${orocos_kdl_LIBRARIES})
target_compile_definitions(geometry_transformation PRIVATE KDL_FOUND=${orocos_kdl_FOUND})


add_executable(eigen_value_eigen_vector src/eigen_value_eigen_vector.cpp)
target_link_libraries (eigen_value_eigen_vector)

add_executable(matrix_condition_numerical_stability src/matrix_condition_numerical_stability.cpp)
target_link_libraries (matrix_condition_numerical_stability)


add_executable(singular_value_decomposition src/singular_value_decomposition.cpp)
target_link_libraries (singular_value_decomposition)

add_executable(sparse_matrices src/sparse_matrices.cpp)
target_link_libraries (sparse_matrices)


add_executable(3_link_planner_robot src/3_link_planner_robot.cpp)
target_link_libraries (3_link_planner_robot)

add_executable(null_space_kernel_rank src/null_space_kernel_rank.cpp)
target_link_libraries (null_space_kernel_rank)

add_executable(solving_system_of_linear_equations src/solving_system_of_linear_equations.cpp)
target_link_libraries (solving_system_of_linear_equations)




#IF(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/sandbox")
#	ADD_SUBDIRECTORY(src/sandbox)
#ENDIF()
