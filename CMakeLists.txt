cmake_minimum_required(VERSION 2.8)
project(Mastering_Eigen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_BUILD_TYPE Debug)



#set(Eigen3_DIR "$ENV{HOME}/usr/share/eigen3/cmake")

find_package (Eigen3 REQUIRED NO_MODULE)

MESSAGE("EIGEN3_FOUND: " ${EIGEN3_FOUND})
MESSAGE("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
MESSAGE("EIGEN3_VERSION: " ${EIGEN3_VERSION})
MESSAGE("EIGEN3_VERSION_STRING: " ${EIGEN3_VERSION_STRING})

include_directories(${EIGEN3_INCLUDE_DIR})



find_package(orocos_kdl)
if(${orocos_kdl_FOUND})
    MESSAGE("orocos_kdl_INCLUDE_DIRS: " ${orocos_kdl_INCLUDE_DIRS})
    MESSAGE("orocos_kdl_LIBRARIES: " ${orocos_kdl_LIBRARIES})
    include_directories(${orocos_kdl_INCLUDE_DIRS})
    MESSAGE("orocos_kdl_LIBRARIES: " ${orocos_kdl_LIBRARIES})
    MESSAGE("orocos_kdl_INCLUDE_DIRS: " ${orocos_kdl_INCLUDE_DIRS})
    MESSAGE("orocos_kdl_LIBRARIES: " ${orocos_kdl_LIBRARIES})
    include_directories(${orocos_kdl_INCLUDE_DIRS})
endif()

find_package(glm)
if(${glm_FOUND})
    MESSAGE("glm_FOUND: " ${glm_FOUND})
    MESSAGE("GLM_LIBRARIES: " ${GLM_LIBRARIES})
    MESSAGE("GLM_INCLUDE_DIRS: " ${GLM_INCLUDE_DIRS})
    add_executable (glm src/glm.cpp)
    target_link_libraries (glm glm::glm)
endif()

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

add_executable(quaternion src/quaternion.cpp)
target_link_libraries (quaternion)

add_executable(rotation_matrices src/rotation_matrices.cpp)
target_link_libraries (rotation_matrices   ${orocos_kdl_LIBRARIES})
target_compile_definitions(rotation_matrices PRIVATE KDL_FOUND=${orocos_kdl_FOUND})

add_executable(angle_axis_rodrigues src/angle_axis_rodrigues.cpp)
target_link_libraries (angle_axis_rodrigues)

add_executable(euler_angle.cpp src/euler_angle.cpp)
target_link_libraries (euler_angle.cpp)

add_executable(eigen_value_eigen_vector src/eigen_value_eigen_vector.cpp)
target_link_libraries (eigen_value_eigen_vector)

add_executable(matrix_condition_numerical_stability src/matrix_condition_numerical_stability.cpp)
target_link_libraries (matrix_condition_numerical_stability)

add_executable(singular_value_decomposition src/singular_value_decomposition.cpp)
target_link_libraries (singular_value_decomposition)

add_executable(sparse_matrices src/sparse_matrices.cpp)
target_link_libraries (sparse_matrices)

add_executable(null_space_kernel_rank src/null_space_kernel_rank.cpp)
target_link_libraries (null_space_kernel_rank)

add_executable(solving_system_of_linear_equations src/solving_system_of_linear_equations.cpp)
target_link_libraries (solving_system_of_linear_equations)

add_executable(numerical_methods src/numerical_methods.cpp)
target_link_libraries (numerical_methods)

add_executable(calculate_range_of_matrix_column_space src/calculate_range_of_matrix_column_space.cpp)
target_link_libraries (calculate_range_of_matrix_column_space)

add_executable(compute_basis_of_nullspace src/compute_basis_of_nullspace.cpp)
target_link_libraries (compute_basis_of_nullspace)

#add_executable(random_number_generation src/random_number_generation.cpp)
#target_link_libraries (random_number_generation)

#IF(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/sandbox")
#	ADD_SUBDIRECTORY(src/sandbox)
#ENDIF()
