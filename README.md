## **This repository is outdated, Please check out my updated repository [Mastering_Eigen](https://github.com/behnamasadi/Mastering_Eigen)**


<br/>
<br/>
<br/>
# [Chapter 1](1_Installation.md)
- [Eigen Introduction and Installation](1_Installation.md)

# [Chapter 2](2_Matrix_Array_Vector_Class.md)
- [Matrix Class](2_Matrix_Array_Vector_Class.md#matrix-class)
- [Vector Class](2_Matrix_Array_Vector_Class.md#vector-class)
- [Array Class](2_Matrix_Array_Vector_Class.md#array-class)
  * [Converting Array to Matrix](2_Matrix_Array_Vector_Class.md#converting-array-to-matrix)
  * [Converting Matrix to Array](2_Matrix_Array_Vector_Class.md#converting-matrix-to-array)
- [Initialization](2_Matrix_Array_Vector_Class.md#initialization)
- [Accessing Elements (Coefficient)](2_Matrix_Array_Vector_Class.md#accessing-elements--coefficient-)
  * [Accessing via parenthesis](2_Matrix_Array_Vector_Class.md#accessing-via-parenthesis)
  * [Accessing via pointer to data](2_Matrix_Array_Vector_Class.md#accessing-via-pointer-to-data)
  * [Row Major Access](2_Matrix_Array_Vector_Class.md#row-major-access)
  * [Accessing a block of data](2_Matrix_Array_Vector_Class.md#accessing-a-block-of-data)
- [Reshaping, Resizing, Slicing](2_Matrix_Array_Vector_Class.md#reshaping--resizing--slicing)
- [Tensor Module](2_Matrix_Array_Vector_Class.md#tensor-module)

# [Chapter 3](3_Matrix_Operations.md)

- [Matrix Arithmetic](3_Matrix_Operations.md#matrix-arithmetic)
  * [Addition/Subtraction Matrices/ Scalar](3_Matrix_Operations.md#addition-subtraction-matrices--scalar)
  * [Scalar Multiplication/ Division](3_Matrix_Operations.md#scalar-multiplication--division)
  * [Multiplication, Dot And Cross Product](3_Matrix_Operations.md#multiplication--dot-and-cross-product)
  * [Transposition and Conjugation](3_Matrix_Operations.md#transposition-and-conjugation)
- [Coefficient-Wise Operations](3_Matrix_Operations.md#coefficient-wise-operations)
  * [Multiplication](3_Matrix_Operations.md#multiplication)
  * [Absolute](3_Matrix_Operations.md#absolute)
  * [Power, Root](3_Matrix_Operations.md#power--root)
  * [Log, Exponential](3_Matrix_Operations.md#log--exponential)
  * [Min, Mix of Two Matrices](3_Matrix_Operations.md#min--mix-of-two-matrices)
  * [Check Matrices Similarity](3_Matrix_Operations.md#check-matrices-similarity)
  * [Finite, Inf, NaN](3_Matrix_Operations.md#finite--inf--nan)
  * [Sinusoidal](3_Matrix_Operations.md#sinusoidal)
  * [Floor, Ceil, Round](3_Matrix_Operations.md#floor--ceil--round)
  * [Masking Elements](3_Matrix_Operations.md#masking-elements)
- [Reductions](3_Matrix_Operations.md#reductions)
  * [Minimum/ Maximum Element In The Matrix](3_Matrix_Operations.md#minimum--maximum-element-in-the-matrix)
  * [Minimum/ Maximum Element Row-wise/Col-wise in the Matrix](3_Matrix_Operations.md#minimum--maximum-element-row-wise-col-wise-in-the-matrix)
  * [Sum Of All Elements](3_Matrix_Operations.md#sum-of-all-elements)
  * [Mean Of The Matrix](3_Matrix_Operations.md#mean-of-the-matrix)
  * [Mean Of The Matrix Row-wise/Col-wise](3_Matrix_Operations.md#mean-of-the-matrix-row-wise-col-wise)
  * [The Trace Of The Matrix](3_Matrix_Operations.md#the-trace-of-the-matrix)
  * [The Multiplication Of All Elements](3_Matrix_Operations.md#the-multiplication-of-all-elements)
  * [Norm 2 of The Matrix](3_Matrix_Operations.md#norm-2-of-the-matrix)
  * [Norm Infinity Of The Matrix](3_Matrix_Operations.md#norm-infinity-of-the-matrix)
  * [Checking If All Elements Are Positive](3_Matrix_Operations.md#checking-if-all-elements-are-positive)
  * [Checking If Any Elements Is Positive](3_Matrix_Operations.md#checking-if-any-elements-is-positive)
  * [Counting Elements](3_Matrix_Operations.md#counting-elements)
  * [Matrix Condition Number](3_Matrix_Operations.md#matrix-condition-number)
  * [Matrix Rank](3_Matrix_Operations.md#matrix-rank)
- [Broadcasting](3_Matrix_Operations.md#broadcasting)


# [Chapter 4](4_Advanced_Eigen_Operations.md)

- [Memory Alignment](4_Advanced_Eigen_Operations.md#memory-alignment)
- [Passing Eigen objects by value to functions](4_Advanced_Eigen_Operations.md#passing-eigen-objects-by-value-to-functions)
- [Aliasing](4_Advanced_Eigen_Operations.md#aliasing)
- [Check Matrix Similarity](4_Advanced_Eigen_Operations.md#check-matrix-similarity)
- [Memory Mapping](4_Advanced_Eigen_Operations.md#memory-mapping)
- [Unary Expression](4_Advanced_Eigen_Operations.md#unary-expression)
- [Eigen Functor](4_Advanced_Eigen_Operations.md#eigen-functor)

# [Chapter 5](5_Dense_Linear_Problems_And_Decompositions.md)

- [Introduction to Linear Equation](5_Dense_Linear_Problems_And_Decompositions.md#introduction-to-linear-equation)
  * [Solution set](5_Dense_Linear_Problems_And_Decompositions.md#solution-set)
  * [Underdetermined System](5_Dense_Linear_Problems_And_Decompositions.md#underdetermined-system)
  * [Overdetermined System](5_Dense_Linear_Problems_And_Decompositions.md#overdetermined-system)
  * [Determined](5_Dense_Linear_Problems_And_Decompositions.md#determined)
  * [Homogeneous vs Non-homogeneous](5_Dense_Linear_Problems_And_Decompositions.md#homogeneous-vs-non-homogeneous)
- [Matrices Decompositions](5_Dense_Linear_Problems_And_Decompositions.md#matrices-decompositions)
  * [QR Decomposition](5_Dense_Linear_Problems_And_Decompositions.md#qr-decomposition)
    + [Square Matrix](5_Dense_Linear_Problems_And_Decompositions.md#square-matrix)
    + [Rectangular Matrix](5_Dense_Linear_Problems_And_Decompositions.md#rectangular-matrix)
    + [Computing the QR Decomposition](5_Dense_Linear_Problems_And_Decompositions.md#computing-the-qr-decomposition)
      - [Gram Schmidt Orthogonalization](5_Dense_Linear_Problems_And_Decompositions.md#gram-schmidt-orthogonalization)
      - [Householder Transformations](5_Dense_Linear_Problems_And_Decompositions.md#householder-transformations)
  * [QL, RQ and LQ Decompositions](5_Dense_Linear_Problems_And_Decompositions.md#ql--rq-and-lq-decompositions)
  * [Cholesky Decomposition](5_Dense_Linear_Problems_And_Decompositions.md#cholesky-decomposition)
  * [LDU Decomposition](5_Dense_Linear_Problems_And_Decompositions.md#ldu-decomposition)
  * [SVD Decomposition](5_Dense_Linear_Problems_And_Decompositions.md#svd-decomposition)
  * [Eigen Value Eigen Vector](5_Dense_Linear_Problems_And_Decompositions.md#eigen-value-eigen-vector)
  * [Basis of Nullspace and Kernel](5_Dense_Linear_Problems_And_Decompositions.md#basis-of-nullspace-and-kernel)
- [Solving Linear Equation](5_Dense_Linear_Problems_And_Decompositions.md#solving-linear-equation)


# [Chapter 6](6_Sparse_Matrices.md)

- [Sparse Matrix Manipulations](6_Sparse_Matrices.md#sparse-matrix-manipulations)
- [Solving Sparse Linear Systems](6_Sparse_Matrices.md#solving-sparse-linear-systems)
- [Matrix Free Solvers](6_Sparse_Matrices.md#matrix-free-solvers)


# [Chapter 7](7_Geometry_Transformation.md)

# [Chapter 8](8_Differentiation.md)
- [Jacobian](8_Differentiation.md#jacobian)
- [Hessian Matrix](8_Differentiation.md#hessian-matrix)
- [Automatic Differentiation](8_Differentiation.md#automatic-differentiation)
- [Numerical Differentiation](8_Differentiation.md#numerical-differentiation)

# [Chapter 9](9_Numerical_Optimization.md)
- [Newton's Method In Optimization](9_Numerical_Optimization.md#newton-s-method-in-optimization)
- [Gauss-Newton Algorithm](9_Numerical_Optimization.md#gauss-newton-algorithm)
    + [Example of Gauss-Newton, Inverse Kinematic Problem](9_Numerical_Optimization.md#example-of-gauss-newton--inverse-kinematic-problem)
- [Curve Fitting](9_Numerical_Optimization.md#curve-fitting)
- [Non Linear Least Squares](9_Numerical_Optimization.md#non-linear-least-squares)
- [Non Linear Regression](9_Numerical_Optimization.md#non-linear-regression)
- [Levenberg Marquardt](9_Numerical_Optimization.md#levenberg-marquardt)



