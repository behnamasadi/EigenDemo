# Chapter 2 Matrix, Array and Vector Class
- [Matrix Class](#matrix-class)
- [Vector Class](#vector-class)
- [Array Class](#array-class)
  * [Converting Array to Matrix](#converting-array-to-matrix)
  * [Converting Matrix to Array](#converting-matrix-to-array)
- [Initialization](#initialization)
- [Accessing Elements (Coefficient)](#accessing-elements--coefficient-)
  * [Accessing via parenthesis](#accessing-via-parenthesis)
  * [Accessing via pointer to data](#accessing-via-pointer-to-data)
  * [Row Major Access](#row-major-access)
  * [Accessing a block of data](#accessing-a-block-of-data)
- [Casting Matrices](#casting-matrices)
- [Reshaping, Resizing, Slicing](#reshaping--resizing--slicing)
- [Tensor Module](#tensor-module)

The runnable code for this chapter is in
[`src/matrix_array_vector.cpp`](src/matrix_array_vector.cpp).

# Matrix Class
The `Matrix` class takes six template parameters, but the first three are
mandatory:

```cpp
Eigen::Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
```

For instance:
```cpp
Eigen::Matrix<double, 2, 1> matrix;
```

If the dimensions are **not** known at compile time, use `Eigen::Dynamic` as the
template parameter, for instance:
```cpp
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
```

Eigen offers many convenience `typedef`s to cover the usual cases. Their names
follow the pattern:
```
Eigen::Matrix{Size}{Type}
Eigen::Vector{Size}{Type}
Eigen::Array{Size}{Type}
```
where `Type` can be:
- `i` for `int`,
- `f` for `float`,
- `d` for `double`,
- `cf` for `complex<float>`,
- `cd` for `complex<double>`.

and `Size` can be `2`, `3`, `4` for fixed-size square matrices, or `X` for
dynamic size.

For example, `Matrix4f` is a `4x4` matrix of floats, defined by Eigen as:
```cpp
typedef Matrix<float, 4, 4> Matrix4f;
```
and a fully dynamic double matrix is:
```cpp
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
```

Here are more examples:

```cpp
Eigen::Matrix4d m;             // 4x4 double

Eigen::Matrix4cd objMatrix4cd; // 4x4 complex<double>

// a is a 3x3 matrix with a static float[9] array of uninitialized coefficients.
Eigen::Matrix3f a;

// b is a dynamic-size matrix whose size is currently 0x0, and whose array of
// coefficients hasn't been allocated at all yet.
Eigen::MatrixXf b;

// A is a 10x15 dynamic-size matrix with allocated but currently uninitialized
// coefficients.
Eigen::MatrixXf A(10, 15);
```

# Vector Class
Vectors are simply matrices with a single column (column vectors) or a single
row (row vectors). For instance:
```cpp
// Vector3f is a fixed-size column vector of 3 floats:
Eigen::Vector3f objVector3f;

// RowVector2i is a fixed-size row vector of 2 ints:
Eigen::RowVector2i objRowVector2i;

// VectorXf is a dynamic-size column vector of floats; here it has size 10:
Eigen::VectorXf objv(10);

// V is a dynamic-size vector of size 30, with allocated but currently
// uninitialized coefficients.
Eigen::VectorXf V(30);
```
You can get/set each row or column of a matrix from a vector:

```cpp
Eigen::Matrix2d mat;
mat << 1, 2,
       3, 4;
Eigen::RowVector2d firstRow = mat.row(0);
Eigen::Vector2d    firstCol = mat.col(0);

firstRow = Eigen::RowVector2d::Random();
firstCol = Eigen::Vector2d::Random();

mat.row(0) = firstRow;
mat.col(0) = firstCol;
```

# Array Class
The `Matrix` class is intended for linear algebra. The `Array` class provides
general-purpose arrays with **coefficient-wise** operations, such as adding a
constant to every coefficient or multiplying two arrays element by element.

```cpp
// ArrayXf
Eigen::Array<float, Eigen::Dynamic, 1> a1;
// Array3f
Eigen::Array<float, 3, 1> a2;
// ArrayXXd
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> a3;
// Array33d
Eigen::Array<double, 3, 3> a4;
```

`Matrix` and `Array` are easily convertible. The `.array()` method gives an
array view of a matrix, and `.matrix()` gives a matrix view of an array.

## Converting Array to Matrix
```cpp
Eigen::Array<double, 4, 4>  array1 = Eigen::Array<double, 4, 4>::Random();
Eigen::Matrix<double, 4, 4> mat1   = array1.matrix();
```

## Converting Matrix to Array
```cpp
Eigen::Matrix<double, 4, 4> mat1   = Eigen::MatrixXd::Random(4, 4);
Eigen::Array<double, 4, 4>  array1 = mat1.array();
```

# Initialization
You can initialize a matrix with the comma initializer (values are filled in
row-major order regardless of storage order):
```cpp
Eigen::Matrix<double, 2, 3> matrix;
matrix << 1, 2, 3,
          4, 5, 6;
```

There are various out-of-the-box APIs for special matrices, for instance:

```cpp
Eigen::Matrix2d rndMatrix;
rndMatrix.setRandom();

Eigen::Matrix2d constantMatrix;
constantMatrix.setConstant(4.3);

Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(6, 6);

Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(3, 3);

Eigen::ArrayXXf table(10, 4);
table.col(0) = Eigen::ArrayXf::LinSpaced(10, 0, 90);
```

# Accessing Elements (Coefficient)
## Accessing via parenthesis
Eigen overloads the parenthesis operator, so you can access elements by row and
column index: `matrix(row, col)`.

All Eigen matrices default to **column-major** storage order. That means the
single-index access `matrix(2)` returns the third element of the first column,
which is the same as `matrix(2, 0)`:

```cpp
Eigen::MatrixXf matrix(4, 4);
matrix << 1,  2,  3,  4,
          5,  6,  7,  8,
          9,  10, 11, 12,
          13, 14, 15, 16;

std::cout << "matrix(2):   " << matrix(2)    << std::endl; // 9
std::cout << "matrix(2,0): " << matrix(2, 0) << std::endl; // 9
```

## Accessing via pointer to data

If you need to access the underlying buffer directly, `matrix.data()` returns a
pointer to the first element:

```cpp
for (int i = 0; i < matrix.size(); i++) {
  std::cout << *(matrix.data() + i) << "  ";
}
```

## Row Major Access

By default Eigen matrices are column-major. To change the storage order, pass
the `Eigen::RowMajor` option as the (optional) fourth template parameter:
```cpp
Eigen::Matrix<double, 4, 4, Eigen::RowMajor> matrixRowMajor;
```
Iterating over `data()` then walks the buffer row by row instead of column by
column.

## Accessing a block of data

You can access a rectangular block of a matrix with
`matrix.block(i, j, p, q)`, which selects a `p x q` block starting at
`(i, j)`:
```cpp
int starting_row          = 1;
int starting_column       = 1;
int number_rows_in_block  = 2;
int number_cols_in_block  = 2;

matrix.block(starting_row, starting_column,
             number_rows_in_block, number_cols_in_block);
```

# Casting Matrices

To mix scalar types in an operation you must `cast<T>()` one operand so both
sides share the same scalar type:

```cpp
Eigen::Matrix<float, 2, 3> matrix_23;
matrix_23 << 1, 2, 3,
             4, 5, 6;
Eigen::Vector3d v_3d(1, 2, 3);

// This would NOT compile: float matrix * double vector.
// Eigen::Matrix<double, 2, 1> wrong = matrix_23 * v_3d;

// Cast the float matrix to double first:
Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
```

# Reshaping, Resizing, Slicing
The current size of a matrix is retrieved with `rows()`, `cols()` and `size()`.
A dynamic-size matrix is resized with `resize()`. `resize()` is a no-op only if
the requested dimensions are identical to the current ones; **otherwise it is
destructive** — the coefficients are not preserved (so resizing from `3x4` to
`6x2` reallocates and loses the old values, even though the total element count
is unchanged).

```cpp
int rows = 3, cols = 4;
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dynamicMatrix;

dynamicMatrix.resize(rows, cols);
dynamicMatrix = Eigen::MatrixXd::Random(rows, cols);
dynamicMatrix.resize(2, 6); // values are not preserved
```

If you want to change the size while **keeping** the existing data, use
`conservativeResize()`:
```cpp
dynamicMatrix.conservativeResize(dynamicMatrix.rows(),
                                 dynamicMatrix.cols() + 1);
dynamicMatrix.col(dynamicMatrix.cols() - 1) = Eigen::Vector2d(1, 4);
```

## Reshaping
Since Eigen 3.4, `reshaped()` returns a view of the same coefficients arranged
with different dimensions (column-major order by default), without copying:
```cpp
Eigen::MatrixXd m(2, 3);
m << 1, 2, 3,
     4, 5, 6;

m.reshaped(3, 2);          // a 3x2 view of the same data
m.reshaped().transpose();  // flatten to a single row: 1 4 2 5 3 6
```

## Slicing
Also since Eigen 3.4, you can index a matrix with `seq()`, `seqN()`, `all`, and
`last` to extract sub-blocks, strided ranges, or whole rows/columns:
```cpp
Eigen::MatrixXd m(4, 4);
m << 1,  2,  3,  4,
     5,  6,  7,  8,
     9,  10, 11, 12,
     13, 14, 15, 16;

m(Eigen::seq(1, 2), Eigen::all);            // rows 1..2, all columns
m(Eigen::seq(0, Eigen::last, 2), Eigen::all); // every other row (stride 2)
m(Eigen::all, Eigen::last);                 // the last column
```

# Tensor Module

Eigen also ships an (unsupported) `Tensor` module for multi-dimensional arrays,
useful when matrices and vectors (rank 2 and rank 1) are not enough. It lives
under `unsupported/Eigen/CXX11/Tensor`:

```cpp
#include <unsupported/Eigen/CXX11/Tensor>

// A rank-3 tensor of size 2 x 3 x 4.
Eigen::Tensor<double, 3> tensor(2, 3, 4);
tensor.setZero();
tensor(0, 1, 2) = 42.0;
```

Being "unsupported" means it is not part of Eigen's stable API and may change
between releases, but it is widely used (for example by TensorFlow's Eigen
backend).

[<< Previous ](1_Intro_Installation.md)  [Home](README.md) [ Next >>](3_Matrix_Operations.md)
