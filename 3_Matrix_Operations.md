#  Chapter 3 Matrix Operations
- [Matrix Arithmetic](#matrix-arithmetic)
  * [Addition/Subtraction Matrices/ Scalar](#addition-subtraction-matrices--scalar)
  * [Scalar Multiplication/ Division](#scalar-multiplication--division)
  * [Multiplication, Dot And Cross Product](#multiplication--dot-and-cross-product)
  * [Transposition and Conjugation](#transposition-and-conjugation)
- [Coefficient-Wise Operations](#coefficient-wise-operations)
  * [Absolute, Power, Root](#absolute-power-root)
  * [Log, Exponential](#log--exponential)
  * [Min, Max of Two Matrices](#min--max-of-two-matrices)
  * [Finite, Inf, NaN](#finite--inf--nan)
  * [Sinusoidal](#sinusoidal)
  * [Floor, Ceil, Round](#floor--ceil--round)
  * [Masking Elements](#masking-elements)
- [Reductions](#reductions)
  * [Minimum/ Maximum Element In The Matrix](#minimum--maximum-element-in-the-matrix)
  * [Minimum/ Maximum Element Row-wise/Col-wise in the Matrix](#minimum--maximum-element-row-wise-col-wise-in-the-matrix)
  * [Sum, Mean, Trace, Product](#sum-mean-trace-product)
  * [Norms](#norms)
  * [All, Any, Count](#all-any-count)
  * [Matrix Rank](#matrix-rank)
- [Matrix Condition Number and Numerical Stability](#matrix-condition-number-and-numerical-stability)
- [Check Matrices Similarity](#check-matrices-similarity)
- [Broadcasting](#broadcasting)

The runnable code for this chapter is in
[`src/matrix_operations.cpp`](src/matrix_operations.cpp),
[`src/matrix_broadcasting.cpp`](src/matrix_broadcasting.cpp),
[`src/matrix_condition_numerical_stability.cpp`](src/matrix_condition_numerical_stability.cpp)
and [`src/check_matrixsimilarity.cpp`](src/check_matrixsimilarity.cpp).

A key distinction throughout this chapter: the `Matrix` class uses **linear
algebra** semantics (so `a * b` is matrix multiplication), while the `Array`
interface (`.array()`) uses **coefficient-wise** semantics (so `a * b`
multiplies element by element).

# Matrix Arithmetic
## Addition/Subtraction Matrices/ Scalar

Matrices of the same size are added and subtracted element by element. Adding a
scalar to *every* element is a coefficient-wise operation, so it goes through
`.array()`:

```cpp
Eigen::Matrix2d a;
a << 1, 2,
     3, 4;

a + b;                  // matrix addition (same-size matrices)
(a.array() + 2).matrix(); // add 2 to every element
```

## Scalar Multiplication/ Division

```cpp
a * 2.5;  // multiply every element by 2.5
a / 2;    // divide every element by 2
```

## Multiplication, Dot And Cross Product

For `Matrix` objects, `*` is matrix multiplication:

```cpp
a * b;  // matrix product
```

The dot and cross products are defined for vectors:

```cpp
Eigen::Vector3d u(1, 2, 3), v(4, 5, 6);
u.dot(v);    // 32
u.cross(v);  // (-3, 6, -3)
```

## Transposition and Conjugation

`transpose()` returns the transpose. For complex matrices, `conjugate()` takes
the complex conjugate and `adjoint()` returns the conjugate transpose:

```cpp
a.transpose();
c.adjoint();   // conjugate transpose of a complex matrix
```

> Note: do not assign a matrix to itself with `m = m.transpose()` — this aliases
> and produces wrong results. Use `m.transposeInPlace()` instead. (Aliasing is
> covered in Chapter 4.)

# Coefficient-Wise Operations

Coefficient-wise math is performed through the `Array` interface. Either declare
an `Array` directly, or call `.array()` on a matrix to get an array view.

## Absolute, Power, Root
```cpp
array1.abs();
array1.square();   // element^2
array1.cube();     // element^3
array1.sqrt();
array1.pow(array2);
array1.pow(scalar);
```

## Log, Exponential
```cpp
array1.log();
array1.log10();
array1.exp();
```

## Min, Max of Two Matrices
Given two arrays of the same size, `.min(.)` / `.max(.)` build the array of the
coefficient-wise minimum / maximum:
```cpp
array1.min(array2);
array1.max(array2);
```

## Finite, Inf, NaN
These return boolean arrays, useful for validating numerical results:
```cpp
array1.isFinite();
array1.isInf();
array1.isNaN();
```

## Sinusoidal
```cpp
array1.sin();   array1.cos();   array1.tan();
array1.asin();  array1.acos();  array1.atan();
array1.sinh();  array1.cosh();  array1.tanh();
```

## Floor, Ceil, Round
```cpp
array1.floor();
array1.ceil();
array1.round();
```

## Masking Elements
`.select()` chooses, element by element, from one of two matrices based on a
boolean condition. Here we replace each element with the corresponding element
of `P` if the value in `R` is below a threshold, otherwise from `Q`:

```cpp
const int rows = 3, cols = 2;
Eigen::MatrixXf R = Eigen::MatrixXf::Random(rows, cols);
Eigen::MatrixXf P = Eigen::MatrixXf::Constant(rows, cols, 1.0);
Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(rows, cols);

float threshold = 0.5f;
Eigen::MatrixXf masked = (R.array() < threshold).select(P, Q); // (R < threshold ? P : Q)
```

# Reductions
A reduction returns a single scalar (or a row/column vector when applied
`colwise()`/`rowwise()`) summarizing a matrix.

## Minimum/ Maximum Element In The Matrix
`minCoeff`/`maxCoeff` can also report the index of the extremum:
```cpp
Eigen::Index row, col;
matrix.minCoeff(&row, &col);
matrix.maxCoeff(&row, &col);
```

## Minimum/ Maximum Element Row-wise/Col-wise in the Matrix
```cpp
matrix.colwise().maxCoeff(); // max of each column -> row vector
matrix.rowwise().mean();     // mean of each row  -> column vector
```

## Sum, Mean, Trace, Product
```cpp
matrix.sum();
matrix.mean();
matrix.trace();  // sum of the diagonal (square matrices)
matrix.prod();   // product of all elements
```

## Norms
```cpp
matrix.norm();                      // Frobenius / L2 norm
matrix.lpNorm<1>();                 // L1 norm
matrix.lpNorm<Eigen::Infinity>();   // max absolute coefficient
```

## All, Any, Count
On a boolean array, `all()`, `any()` and `count()` reduce it to a single answer:
```cpp
(matrix.array() > 0).all();   // are all elements > 0?
(matrix.array() > 5).any();   // is any element > 5?
(matrix.array() > 3).count(); // how many elements > 3?
```

## Matrix Rank
The column rank of a matrix is the maximal number of linearly independent
columns; the row rank is the dimension of the row space. It can be proven that
**column rank = row rank**, simply called the rank.

A matrix has *full rank* if its rank equals the largest possible value for its
dimensions — the lesser of the number of rows and columns. Otherwise it is
*rank-deficient*.

The most robust way to obtain the rank numerically is through a
rank-revealing decomposition such as full-pivot LU or SVD:

```cpp
Eigen::MatrixXd m(3, 3);
m << 1, 2, 3,
     2, 4, 6,   // = 2 * row 0, so the matrix is rank-deficient
     1, 0, 1;

Eigen::FullPivLU<Eigen::MatrixXd> lu(m);
std::cout << "rank: " << lu.rank() << std::endl; // 2
```

# Matrix Condition Number and Numerical Stability

The condition number (in the 2-norm) is the ratio of the largest to the
smallest singular value. A large condition number means the matrix is
**ill-conditioned**: small perturbations in the input can cause large changes in
the solution of a linear system, so the result loses numerical precision.

```cpp
double conditionNumber(const Eigen::MatrixXd &m) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(m);
  const auto &sv = svd.singularValues();
  return sv(0) / sv(sv.size() - 1);
}
```

For the identity matrix the condition number is `1` (perfectly conditioned),
while for the near-singular matrix `[[1, 1], [1, 1.0001]]` it is about `40002`.

# Check Matrices Similarity

Floating-point matrices should **never** be compared with `==`, because rounding
makes exact equality unreliable. Use `isApprox()`, which tests equality up to a
relative precision, or compare the norm of the difference against a tolerance:

```cpp
Eigen::Vector3d a(1.0, 2.0, 3.0);
Eigen::Vector3d b(1.0, 2.0, 3.0 + 1e-12);

a == b;                  // false (exact comparison)
a.isApprox(b);           // true  (default precision)
a.isApprox(b, 1e-6);     // true
(a - b).norm() < 1e-6;   // true
```

# Broadcasting

Broadcasting replicates a vector along the rows or columns of a matrix so it can
be combined with the whole matrix at once (similar to NumPy broadcasting). Use
`.colwise()` to broadcast a column vector across columns, or `.rowwise()` to
broadcast a row vector across rows:

```cpp
Eigen::MatrixXf mat(2, 4);
mat << 1, 2, 6, 9,
       3, 1, 7, 2;

Eigen::VectorXf v(2);
v << 0, 1;
mat.colwise() + v;   // add v to every column

Eigen::RowVectorXf w(4);
w << 0, 1, 2, 3;
mat.rowwise() + w;   // add w to every row
```

A classic application is finding the column nearest to a target vector:

```cpp
Eigen::Index index;
(mat.colwise() - target).colwise().squaredNorm().minCoeff(&index);
```

[<< Previous ](2_Matrix_Array_Vector_Class.md)  [Home](README.md)  [ Next >>](4_Advanced_Eigen_Operations.md)
