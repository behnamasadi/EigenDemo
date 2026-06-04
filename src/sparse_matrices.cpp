// Chapter 6: Sparse matrices.
// Demonstrates the Compressed Sparse Row (CSR) storage layout and solving a
// sparse linear system.
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

// Compressed Sparse Row: a sparse matrix is stored with three arrays
//   - values:    the non-zero coefficients, in row-major order
//   - innerIndex: the column index of each non-zero  (length = nnz)
//   - outerIndex: where each row starts in the arrays (length = rows + 1)
void compressedSparseRow() {
  /*
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
  */
  const int rows = 8;
  const int cols = 9;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dense_mat(rows, cols);

  dense_mat << 0, 0, 0, 0, 0, 0, 0, 3, 0,
               0, 0, 8, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0,
               4, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 2, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 6, 0, 0, 0, 0, 0,
               0, 9, 0, 0, 5, 0, 0, 0, 0;

  // sparseView() keeps only the non-zeros. A tolerance can be given via
  // dense_mat.sparseView(epsilon, reference).
  Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_mat =
      dense_mat.sparseView();

  std::cout << "rows: " << sparse_mat.rows()
            << ", cols: " << sparse_mat.cols()
            << ", non-zeros: " << sparse_mat.nonZeros() << "\n";

  // The values array has one entry per non-zero.
  std::cout << "values (" << sparse_mat.nonZeros() << "): ";
  const double *valuePtr = sparse_mat.valuePtr();
  for (int i = 0; i < sparse_mat.nonZeros(); i++)
    std::cout << valuePtr[i] << " ";
  std::cout << "\n";

  // For a row-major matrix the inner index is the COLUMN index of each
  // non-zero; this array also has length nnz.
  std::cout << "COL_INDEX (" << sparse_mat.nonZeros() << "): ";
  const int *innerPtr = sparse_mat.innerIndexPtr();
  for (int i = 0; i < sparse_mat.nonZeros(); i++)
    std::cout << innerPtr[i] << " ";
  std::cout << "\n";

  // The outer index points to the start of each row; it has rows + 1 entries
  // (the last one equals nnz).
  std::cout << "ROW_INDEX (" << sparse_mat.outerSize() + 1 << "): ";
  const int *outerPtr = sparse_mat.outerIndexPtr();
  for (int i = 0; i < sparse_mat.outerSize() + 1; i++)
    std::cout << outerPtr[i] << " ";
  std::cout << "\n";
}

// Solving a sparse linear system A x = b. We build a small symmetric
// positive-definite matrix from a list of triplets (i, j, value) and solve it
// with a direct Cholesky factorization.
void solveSparseSystem() {
  std::cout << "\n----- Solving a sparse system A x = b -----\n";
  typedef Eigen::SparseMatrix<double> SpMat;
  typedef Eigen::Triplet<double> T;

  const int n = 3;
  std::vector<T> coefficients = {
      T(0, 0, 4), T(0, 1, 1),
      T(1, 0, 1), T(1, 1, 3), T(1, 2, 1),
      T(2, 1, 1), T(2, 2, 2),
  };

  SpMat A(n, n);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  Eigen::VectorXd b(n);
  b << 1, 2, 3;

  Eigen::SimplicialLDLT<SpMat> solver(A);
  if (solver.info() != Eigen::Success) {
    std::cerr << "decomposition failed\n";
    return;
  }
  Eigen::VectorXd x = solver.solve(b);

  std::cout << "A =\n" << Eigen::MatrixXd(A) << "\n";
  std::cout << "b = " << b.transpose() << "\n";
  std::cout << "x = " << x.transpose() << "\n";
  std::cout << "residual ||Ax - b|| = " << (A * x - b).norm() << "\n";
}

int main() {
  compressedSparseRow();
  solveSparseSystem();
  return 0;
}
