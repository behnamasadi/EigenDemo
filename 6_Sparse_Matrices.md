# Chapter 6 Sparse Matrices
- [Sparse Matrix Manipulations](#sparse-matrix-manipulations)
  * [Compressed Sparse Row](#compressed-sparse-row)
- [Solving Sparse Linear Systems](#solving-sparse-linear-systems)
- [Matrix Free Solvers](#matrix-free-solvers)

The runnable code for this chapter is in
[`src/sparse_matrices.cpp`](src/sparse_matrices.cpp).

# Sparse Matrix Manipulations

A *sparse* matrix is one in which most entries are zero. Storing and operating
on only the non-zero entries saves a great deal of memory and computation.
Eigen provides `Eigen::SparseMatrix<Scalar>` for this, stored by default in
**column-major** order (use the `Eigen::RowMajor` option for row-major).

The most efficient way to build a sparse matrix is from a list of *triplets*
`(row, col, value)`:

```cpp
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

std::vector<T> coefficients = { T(0,0,4), T(0,1,1), T(1,1,3) /* ... */ };
SpMat A(n, n);
A.setFromTriplets(coefficients.begin(), coefficients.end());
```

You can also obtain a sparse matrix from a dense one with `sparseView()`
(optionally with a tolerance: `dense.sparseView(epsilon, reference)`).

## Compressed Sparse Row

Internally Eigen stores a sparse matrix in **compressed** form using three
arrays. For a row-major matrix this is the Compressed Sparse Row (CSR) layout:

- **values** — the non-zero coefficients in row-major order (length = number of
  non-zeros, `nnz`).
- **inner index** (`innerIndexPtr`) — the **column** index of each non-zero
  (length = `nnz`).
- **outer index** (`outerIndexPtr`) — the position in the arrays where each row
  begins (length = `rows + 1`; the last entry equals `nnz`).

For the 8×9 matrix in the source:

```
    0 1 2 3 4 5 6 7 8
   ┌                 ┐
 0 |0 0 0 0 0 0 0 3 0|
 1 |0 0 8 0 0 1 0 0 0|
 2 |0 0 0 0 0 0 0 0 0|
 3 |4 0 0 0 0 0 0 0 0|
 4 |0 0 0 0 0 0 0 0 0|
 5 |0 0 2 0 0 0 0 0 0|
 6 |0 0 0 6 0 0 0 0 0|
 7 |0 9 0 0 5 0 0 0 0|
   └                 ┘
```

the three arrays are:

```
rows: 8, cols: 9, non-zeros: 8
values    (8): 3 8 1 4 2 6 9 5
COL_INDEX (8): 7 2 5 0 2 3 1 4
ROW_INDEX (9): 0 1 3 3 4 4 5 6 8
```

Reading `ROW_INDEX`: row 0 occupies entries `[0,1)` of the arrays (the single
value `3` in column `7`); row 1 occupies `[1,3)` (values `8,1` in columns
`2,5`); row 2 is empty because `ROW_INDEX[2] == ROW_INDEX[3] == 3`; and so on.

> Note: the inner-index array has length `nnz` and the outer-index array has
> length `rows + 1` (not `innerSize()`/`outerSize()`), so iterate over them with
> those bounds.

# Solving Sparse Linear Systems

Eigen offers both direct and iterative sparse solvers. They all share the same
`compute()` / `solve()` interface:

| Solver | Matrix requirement |
| --- | --- |
| `SimplicialLLT` / `SimplicialLDLT` | symmetric positive-definite |
| `SparseLU` | general square |
| `SparseQR` | general (least squares) |
| `ConjugateGradient` (iterative) | symmetric positive-definite |
| `BiCGSTAB` (iterative) | general square |

Example with a direct Cholesky (LDLᵀ) factorization of a small SPD system:

```cpp
SpMat A(n, n);
A.setFromTriplets(coefficients.begin(), coefficients.end());

Eigen::SimplicialLDLT<SpMat> solver(A);
if (solver.info() != Eigen::Success) { /* decomposition failed */ }
Eigen::VectorXd x = solver.solve(b);
```

For the system

```
A =          b = (1 2 3)
4 1 0
1 3 1
0 1 2
```

the solver returns `x = (0.2222, 0.1111, 1.4444)` with residual
`||Ax - b|| ≈ 4.4e-16`.

# Matrix Free Solvers

Iterative solvers (`ConjugateGradient`, `BiCGSTAB`, ...) only ever need the
matrix through the product `A * x`. This means you don't have to store `A`
explicitly: you can supply a *matrix-free* linear operator that implements that
product. This is useful when the matrix is huge or only defined implicitly (for
example a discretized differential operator). Eigen supports this by
specializing `Eigen::internal::traits` for a custom type that wraps your
operator and plugging it into an iterative solver. See the Eigen
[matrix-free solver example](https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html).

[<< Previous ](5_Dense_Linear_Problems_And_Decompositions.md)  [Home](README.md)  [ Next >>](7_Geometry_Transformation.md)
