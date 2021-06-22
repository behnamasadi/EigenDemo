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
target_link_libraries (main ${EIGEN3_LIBRARIES}  )


add_executable(non_linear_regression src/non_linear_regression.cpp)
target_link_libraries (non_linear_regression ${EIGEN3_LIBRARIES}  )

add_executable(non_linear_least_squares src/non_linear_least_squares.cpp)
target_link_libraries (non_linear_least_squares ${EIGEN3_LIBRARIES}  )

add_executable(numerical_diff src/numerical_diff.cpp)
target_link_libraries (numerical_diff ${EIGEN3_LIBRARIES}  )

add_executable(levenberg_marquardt src/levenberg_marquardt.cpp)
target_link_libraries (levenberg_marquardt ${EIGEN3_LIBRARIES}  )

add_executable(eigen_functor src/eigen_functor.cpp)
target_link_libraries (eigen_functor ${EIGEN3_LIBRARIES}  )

add_executable(auto_diff src/auto_diff.cpp)
target_link_libraries (auto_diff ${EIGEN3_LIBRARIES}  )


add_executable(matrix_array_vector src/matrix_array_vector.cpp)
target_link_libraries (matrix_array_vector ${EIGEN3_LIBRARIES}  )


add_executable(unaryExpr src/unaryExpr.cpp)
target_link_libraries (unaryExpr ${EIGEN3_LIBRARIES}  )

add_executable(memory_mapping src/memory_mapping.cpp)
target_link_libraries (memory_mapping ${EIGEN3_LIBRARIES}  )

add_executable(matrix_broadcasting src/matrix_broadcasting.cpp)
target_link_libraries (matrix_broadcasting ${EIGEN3_LIBRARIES}  )

add_executable(matrix_decomposition src/matrix_decomposition.cpp)
target_link_libraries (matrix_decomposition ${EIGEN3_LIBRARIES}  )

add_executable(gram_schmidt_orthogonalization src/gram_schmidt_orthogonalization.cpp)
target_link_libraries (gram_schmidt_orthogonalization ${EIGEN3_LIBRARIES}  )


add_executable(check_matrixsimilarity src/check_matrixsimilarity.cpp)
target_link_libraries (check_matrixsimilarity ${EIGEN3_LIBRARIES}  )

add_executable(geometry_transformation src/geometry_transformation.cpp)
target_link_libraries (geometry_transformation ${EIGEN3_LIBRARIES}  ${orocos_kdl_LIBRARIES})
target_compile_definitions(geometry_transformation PRIVATE KDL_FOUND=${orocos_kdl_FOUND})


add_executable(eigen_value_eigen_vector src/eigen_value_eigen_vector.cpp)
target_link_libraries (eigen_value_eigen_vector ${EIGEN3_LIBRARIES}  )


add_executable(matrix_rank src/matrix_rank.cpp)
target_link_libraries (matrix_rank ${EIGEN3_LIBRARIES}  )

add_executable(matrix_condition src/matrix_condition.cpp)
target_link_libraries (matrix_condition ${EIGEN3_LIBRARIES}  )


add_executable(singular_value_decomposition src/singular_value_decomposition.cpp)
target_link_libraries (singular_value_decomposition ${EIGEN3_LIBRARIES}  )

add_executable(sparse_matrices src/sparse_matrices.cpp)
target_link_libraries (sparse_matrices ${EIGEN3_LIBRARIES}  )



#IF(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/sandbox")
#	ADD_SUBDIRECTORY(src/sandbox)
#ENDIF()
