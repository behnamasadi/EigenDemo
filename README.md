# Linear Algebra With Eigen and C++

**CI**  
[![Linux GCC/Clang](https://github.com/behnamasadi/EigenDemo/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/behnamasadi/EigenDemo/actions/workflows/ci.yml)
[![Windows MSVC](https://github.com/behnamasadi/EigenDemo/actions/workflows/windows.yml/badge.svg?branch=master)](https://github.com/behnamasadi/EigenDemo/actions/workflows/windows.yml)
[![Links](https://github.com/behnamasadi/EigenDemo/actions/workflows/links.yml/badge.svg?branch=master)](https://github.com/behnamasadi/EigenDemo/actions/workflows/links.yml)
[![CodeQL](https://github.com/behnamasadi/EigenDemo/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/behnamasadi/EigenDemo/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/behnamasadi/EigenDemo/badge)](https://scorecard.dev/viewer/?uri=github.com/behnamasadi/EigenDemo)

**Stack**  
![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white)
[![Eigen 3.4 | 5](https://img.shields.io/badge/Eigen-3.4%20%7C%205-2E6DB4)](https://eigen.tuxfamily.org/)
![CMake](https://img.shields.io/badge/CMake-3.21%2B-064F8C?logo=cmake&logoColor=white)
![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)

**Repository**  
[![License](https://img.shields.io/github/license/behnamasadi/EigenDemo)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/behnamasadi/EigenDemo)](https://github.com/behnamasadi/EigenDemo/commits/master)
[![Commit activity](https://img.shields.io/github/commit-activity/y/behnamasadi/EigenDemo)](https://github.com/behnamasadi/EigenDemo/graphs/commit-activity)
[![Contributors](https://img.shields.io/github/contributors/behnamasadi/EigenDemo)](https://github.com/behnamasadi/EigenDemo/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/behnamasadi/EigenDemo)](https://github.com/behnamasadi/EigenDemo/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/behnamasadi/EigenDemo)](https://github.com/behnamasadi/EigenDemo/pulls)
![Top language](https://img.shields.io/github/languages/top/behnamasadi/EigenDemo)
![Code size](https://img.shields.io/github/languages/code-size/behnamasadi/EigenDemo)
[![Stars](https://img.shields.io/github/stars/behnamasadi/EigenDemo?style=social)](https://github.com/behnamasadi/EigenDemo/stargazers)
[![Forks](https://img.shields.io/github/forks/behnamasadi/EigenDemo?style=social)](https://github.com/behnamasadi/EigenDemo/network/members)

This repository contains my tutorials on mastering Matrix operation and numerical optimization with Eigen and C++.
Every chapter below is a written tutorial paired with a self-contained, compilable example in [`src/`](src) — from
basic matrix arithmetic up to SVD-based camera calibration, point-cloud registration and sparse SLAM pose-graph solving.

## Build and run

Requires a C++23 compiler (GCC 13+, Clang 17+, Apple Clang 15+, MSVC 2022), [CMake](https://cmake.org/) >= 3.21, [Ninja](https://ninja-build.org/) and
[Eigen](https://eigen.tuxfamily.org/) 3.4 or 5 (`sudo apt install libeigen3-dev` on Debian/Ubuntu).

```bash
git clone https://github.com/behnamasadi/EigenDemo.git
cd EigenDemo
cmake --preset ninja-multi
cmake --build build --config Release
```

The binaries land in `build/Release/`, one per example, so you can run any topic directly:

```bash
./build/Release/singular_value_decomposition
./build/Release/quaternion
./build/Release/slam_pose_graph
```

The following is the outline of this repository:

# [Chapter 1 Introduction and Installation](1_Intro_Installation.md)
- [About Eigen](1_Intro_Installation.md#about-eigen)
- [Installation](1_Intro_Installation.md#installation)
- [Adding Eigen to Your Project](1_Intro_Installation.md#adding-eigen-to-your-project)
- [Your First Eigen Program](1_Intro_Installation.md#your-first-eigen-program)

# [Chapter 2 Matrix, Array and Vector Class](2_Matrix_Array_Vector_Class.md)
- [Matrix Class](2_Matrix_Array_Vector_Class.md#matrix-class)
- [Vector Class](2_Matrix_Array_Vector_Class.md#vector-class)
- [Array Class](2_Matrix_Array_Vector_Class.md#array-class)
 - [Initialization](2_Matrix_Array_Vector_Class.md#initialization)
- [Accessing Elements (Coefficient)](2_Matrix_Array_Vector_Class.md#accessing-elements-coefficient)
- [Casting Matrices](2_Matrix_Array_Vector_Class.md#casting-matrices)
- [Reshaping, Resizing, Slicing](2_Matrix_Array_Vector_Class.md#reshaping-resizing-slicing)
  * [Reshaping](2_Matrix_Array_Vector_Class.md#reshaping)
  * [Slicing](2_Matrix_Array_Vector_Class.md#slicing)
- [Tensor Module](2_Matrix_Array_Vector_Class.md#tensor-module)


# [Chapter 3 Matrix Operations](3_Matrix_Operations.md)

- [Matrix Arithmetic](3_Matrix_Operations.md#matrix-arithmetic)
  * [Addition/Subtraction Matrices/ Scalar](3_Matrix_Operations.md#additionsubtraction-matrices-scalar)
  * [Scalar Multiplication/ Division](3_Matrix_Operations.md#scalar-multiplication-division)
  * [Multiplication, Dot And Cross Product](3_Matrix_Operations.md#multiplication-dot-and-cross-product)
  * [Transposition and Conjugation](3_Matrix_Operations.md#transposition-and-conjugation)
- [Coefficient-Wise Operations](3_Matrix_Operations.md#coefficient-wise-operations)
  * [Absolute, Power, Root](3_Matrix_Operations.md#absolute-power-root)
  * [Log, Exponential](3_Matrix_Operations.md#log-exponential)
  * [Min, Max of Two Matrices](3_Matrix_Operations.md#min-max-of-two-matrices)
  * [Finite, Inf, NaN](3_Matrix_Operations.md#finite-inf-nan)
  * [Sinusoidal](3_Matrix_Operations.md#sinusoidal)
  * [Floor, Ceil, Round](3_Matrix_Operations.md#floor-ceil-round)
  * [Masking Elements](3_Matrix_Operations.md#masking-elements)
- [Reductions](3_Matrix_Operations.md#reductions)
  * [Minimum/ Maximum Element In The Matrix](3_Matrix_Operations.md#minimum-maximum-element-in-the-matrix)
  * [Minimum/ Maximum Element Row-wise/Col-wise in the Matrix](3_Matrix_Operations.md#minimum-maximum-element-row-wisecol-wise-in-the-matrix)
  * [Sum, Mean, Trace, Product](3_Matrix_Operations.md#sum-mean-trace-product)
  * [Norms](3_Matrix_Operations.md#norms)
  * [All, Any, Count](3_Matrix_Operations.md#all-any-count)
  * [Matrix Rank](3_Matrix_Operations.md#matrix-rank)
- [Matrix Condition Number and Numerical Stability](3_Matrix_Operations.md#matrix-condition-number-and-numerical-stability)
- [Check Matrices Similarity](3_Matrix_Operations.md#check-matrices-similarity)
- [Broadcasting](3_Matrix_Operations.md#broadcasting)


# [Chapter 4 Advanced Eigen Operations](4_Advanced_Eigen_Operations.md)

- [Memory Alignment](4_Advanced_Eigen_Operations.md#memory-alignment)
- [Passing Eigen objects by value to functions](4_Advanced_Eigen_Operations.md#passing-eigen-objects-by-value-to-functions)
- [Aliasing](4_Advanced_Eigen_Operations.md#aliasing)
- [Memory Mapping](4_Advanced_Eigen_Operations.md#memory-mapping)
  * [Eigen matrix from std::vector](4_Advanced_Eigen_Operations.md#eigen-matrix-from-stdvector)
- [Unary Expression](4_Advanced_Eigen_Operations.md#unary-expression)
- [Eigen Functor](4_Advanced_Eigen_Operations.md#eigen-functor)

# [Chapter 5 Dense Linear Problems And Decompositions](5_Dense_Linear_Problems_And_Decompositions.md)

- [1. Vector Space](5_Dense_Linear_Problems_And_Decompositions.md#1-vector-space)
- [2. Linear Equation](5_Dense_Linear_Problems_And_Decompositions.md#2-linear-equation)
- [3. Solving Linear Equation](5_Dense_Linear_Problems_And_Decompositions.md#3-solving-linear-equation)
- [4. Matrices Decompositions](5_Dense_Linear_Problems_And_Decompositions.md#4-matrices-decompositions)
- [5. Linear Map](5_Dense_Linear_Problems_And_Decompositions.md#5-linear-map)
- [6. Span](5_Dense_Linear_Problems_And_Decompositions.md#6-span)
- [7. Subspace](5_Dense_Linear_Problems_And_Decompositions.md#7-subspace)
- [8. Range of a Matrix](5_Dense_Linear_Problems_And_Decompositions.md#8-range-of-a-matrix)
- [9. Basis](5_Dense_Linear_Problems_And_Decompositions.md#9-basis)
- [10. Rank of Matrix](5_Dense_Linear_Problems_And_Decompositions.md#10-rank-of-matrix)
- [11. Dimension of the Column Space](5_Dense_Linear_Problems_And_Decompositions.md#11-dimension-of-the-column-space)
- [12. Null Space (Kernel)](5_Dense_Linear_Problems_And_Decompositions.md#12-null-space-kernel)
- [13. Nullity](5_Dense_Linear_Problems_And_Decompositions.md#13-nullity)
- [14. Rank-nullity Theorem](5_Dense_Linear_Problems_And_Decompositions.md#14-rank-nullity-theorem)
- [15. The Determinant of The Matrix](5_Dense_Linear_Problems_And_Decompositions.md#15-the-determinant-of-the-matrix)
- [16. Finding The Inverse of The Matrix](5_Dense_Linear_Problems_And_Decompositions.md#16-finding-the-inverse-of-the-matrix)
- [17. The Fundamental Theorem of Linear Algebra](5_Dense_Linear_Problems_And_Decompositions.md#17-the-fundamental-theorem-of-linear-algebra)
- [18. Permutation Matrix](5_Dense_Linear_Problems_And_Decompositions.md#18-permutation-matrix)
- [19. Augmented Matrix](5_Dense_Linear_Problems_And_Decompositions.md#19-augmented-matrix)

# [Chapter 6 Sparse Matrices](6_Sparse_Matrices.md)

- [Sparse Matrix Manipulations](6_Sparse_Matrices.md#sparse-matrix-manipulations)
  * [Compressed Sparse Row](6_Sparse_Matrices.md#compressed-sparse-row)
- [Solving Sparse Linear Systems](6_Sparse_Matrices.md#solving-sparse-linear-systems)
- [Matrix Free Solvers](6_Sparse_Matrices.md#matrix-free-solvers)


# [Chapter 7 Geometry Transformation](7_Geometry_Transformation.md)
- [1. Euler Angles](7_Geometry_Transformation.md#1-euler-angles)
- [2. Global References and Local Tangent Plane Coordinates](7_Geometry_Transformation.md#2-global-references-and-local-tangent-plane-coordinates)
- [3. Axis-angle Representation](7_Geometry_Transformation.md#3-axis-angle-representation)
- [4. Quaternions](7_Geometry_Transformation.md#4-quaternions)
- [5. Conversion between different representations](7_Geometry_Transformation.md#5-conversion-between-different-representations)

# [Chapter 8 Differentiation](8_Differentiation.md)
- [Jacobian](8_Differentiation.md#jacobian)
- [Hessian Matrix](8_Differentiation.md#hessian-matrix)
- [Automatic Differentiation](8_Differentiation.md#automatic-differentiation)
- [Numerical Differentiation](8_Differentiation.md#numerical-differentiation)

# [Chapter 9 Numerical Optimization](9_Numerical_Optimization.md)
- [Newton's Method In Optimization](9_Numerical_Optimization.md#newtons-method-in-optimization)
- [Gauss-Newton Algorithm](9_Numerical_Optimization.md#gauss-newton-algorithm)
    + [Example of Gauss-Newton, Inverse Kinematic Problem](9_Numerical_Optimization.md#example-of-gauss-newton-inverse-kinematic-problem)
- [Quasi-Newton Method](9_Numerical_Optimization.md#quasi-newton-method)
- [Curve Fitting](9_Numerical_Optimization.md#curve-fitting)
- [Non Linear Least Squares](9_Numerical_Optimization.md#non-linear-least-squares)
- [Non Linear Regression](9_Numerical_Optimization.md#non-linear-regression)
- [Levenberg Marquardt](9_Numerical_Optimization.md#levenberg-marquardt)

# [Chapter 10 Linear Algebra in Robotics](10_Linear_Algebra_in_Robotics.md)
- [Why decompositions matter](10_Linear_Algebra_in_Robotics.md#why-decompositions-matter)
- [Inverse Kinematics — the pseudo-inverse (SVD)](10_Linear_Algebra_in_Robotics.md#inverse-kinematics--the-pseudo-inverse-svd)
- [Camera Calibration — DLT (SVD), projection decomposition (QR), Zhang (Cholesky)](10_Linear_Algebra_in_Robotics.md#camera-calibration--dlt-svd-projection-decomposition-qr-zhang-cholesky)
- [SLAM — least squares, QR vs Cholesky, and sparsity](10_Linear_Algebra_in_Robotics.md#slam--least-squares-qr-vs-cholesky-and-sparsity)
- [Point-Cloud Registration — Kabsch / Umeyama (SVD)](10_Linear_Algebra_in_Robotics.md#point-cloud-registration--kabsch--umeyama-svd)
- [PCA & Plane Fitting — eigendecomposition](10_Linear_Algebra_in_Robotics.md#pca--plane-fitting--eigendecomposition)
- [Further reading & related projects](10_Linear_Algebra_in_Robotics.md#further-reading--related-projects)



