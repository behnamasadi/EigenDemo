# ----------------------------------------------------------------------------
# Mastering Eigen — tutorial examples
#
# The "VERSION min...max" range form tells CMake to enable all policies up to
# the newest version it supports (here 3.31) while still configuring on 3.21+.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21...3.31)

project(Mastering_Eigen
    VERSION 1.0.0
    DESCRIPTION "Tutorials on linear algebra and numerical optimization with Eigen and C++"
    HOMEPAGE_URL "https://github.com/behnamasadi/EigenDemo"
    LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Language standard (applied per target via target_compile_features below).
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clangd / editor tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Collect all binaries under build/bin instead of scattering them per target.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# ----------------------------------------------------------------------------
# Default build type for single-config generators (Make, Ninja).
# Multi-config generators (Ninja Multi-Config, Visual Studio) ignore this.
# ----------------------------------------------------------------------------
get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _isMultiConfig AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Found Eigen3 ${Eigen3_VERSION}: ${EIGEN3_INCLUDE_DIR}")

# Optional dependencies — examples that need them are only built when present.
find_package(glm QUIET)
find_package(orocos_kdl QUIET)
message(STATUS "glm found:         ${glm_FOUND}")
message(STATUS "orocos_kdl found:  ${orocos_kdl_FOUND}")

# ----------------------------------------------------------------------------
# Common warning flags, exposed through an INTERFACE target so every example
# opts in by linking it. Warnings only — never -Werror, so the tutorials build
# cleanly everywhere.
# ----------------------------------------------------------------------------
add_library(project_warnings INTERFACE)
if(MSVC)
    target_compile_options(project_warnings INTERFACE /W3)
else()
    target_compile_options(project_warnings INTERFACE -Wall -Wextra)
endif()

# ----------------------------------------------------------------------------
# Helper: register one self-contained example, src/<name>.cpp -> executable.
# ----------------------------------------------------------------------------
function(add_eigen_example name)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} PRIVATE Eigen3::Eigen project_warnings)
    target_compile_features(${name} PRIVATE cxx_std_17)
endfunction()

set(EIGEN_EXAMPLES
    hello_eigen
    main
    matrix_array_vector
    matrix_operations
    matrix_broadcasting
    matrix_decomposition
    matrix_condition_numerical_stability
    unaryExpr
    eigen_functor
    memory_mapping
    check_matrixsimilarity
    gram_schmidt_orthogonalization
    eigen_value_eigen_vector
    singular_value_decomposition
    null_space_kernel_rank
    solving_system_of_linear_equations
    calculate_range_of_matrix_column_space
    compute_basis_of_nullspace
    sparse_matrices
    geometry_transformation
    quaternion
    angle_axis_rodrigues
    euler_angle
    numerical_diff
    numerical_methods
    auto_diff
    non_linear_least_squares
    non_linear_regression
    levenberg_marquardt
    3_link_planner_robot
    camera_calibration_dlt
    slam_pose_graph
)

foreach(example IN LISTS EIGEN_EXAMPLES)
    add_eigen_example(${example})
endforeach()

# ----------------------------------------------------------------------------
# rotation_matrices: optionally uses orocos_kdl. KDL_FOUND is always defined
# (0 or 1) so the `#if KDL_FOUND == 1` guards compile either way.
# ----------------------------------------------------------------------------
add_eigen_example(rotation_matrices)
if(orocos_kdl_FOUND)
    target_link_libraries(rotation_matrices PRIVATE ${orocos_kdl_LIBRARIES})
    target_include_directories(rotation_matrices PRIVATE ${orocos_kdl_INCLUDE_DIRS})
    target_compile_definitions(rotation_matrices PRIVATE KDL_FOUND=1)
else()
    target_compile_definitions(rotation_matrices PRIVATE KDL_FOUND=0)
endif()

# ----------------------------------------------------------------------------
# glm example: only built when glm is available (does not use Eigen).
# ----------------------------------------------------------------------------
if(glm_FOUND)
    add_executable(glm_example src/glm.cpp)
    target_link_libraries(glm_example PRIVATE glm::glm project_warnings)
    target_compile_features(glm_example PRIVATE cxx_std_17)
endif()
