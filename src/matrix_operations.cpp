// Chapter 3: Matrix Operations.
// Demonstrates matrix arithmetic, coefficient-wise (array) operations, and
// reductions. Each topic lives in its own function and is called from main().
#include <Eigen/Dense>
#include <iostream>

void matrixArithmetic() {
  std::cout << "////////////////// Matrix Arithmetic //////////////////\n";

  Eigen::Matrix2d a;
  a << 1, 2,
       3, 4;
  Eigen::Matrix2d b;
  b << 5, 6,
       7, 8;

  std::cout << "a + b =\n" << a + b << "\n";
  std::cout << "a - b =\n" << a - b << "\n";

  // Adding/subtracting a scalar is a coefficient-wise (array) operation.
  std::cout << "a (with 2 added to every element) =\n"
            << (a.array() + 2).matrix() << "\n";

  // Scalar multiplication / division.
  std::cout << "a * 2.5 =\n" << a * 2.5 << "\n";
  std::cout << "a / 2 =\n" << a / 2 << "\n";

  // Matrix multiplication (not coefficient-wise).
  std::cout << "a * b =\n" << a * b << "\n";

  // Transpose and (for complex matrices) conjugate / adjoint.
  std::cout << "a^T =\n" << a.transpose() << "\n";

  Eigen::MatrixXcd c(2, 2);
  c << std::complex<double>(1, 1), std::complex<double>(2, -1),
       std::complex<double>(0, 1), std::complex<double>(3, 2);
  std::cout << "c.adjoint() (conjugate transpose) =\n" << c.adjoint() << "\n";
}

void dotAndCrossProduct() {
  std::cout << "////////////////// Dot and Cross Product //////////////////\n";
  Eigen::Vector3d u(1, 2, 3);
  Eigen::Vector3d v(4, 5, 6);

  std::cout << "u . v = " << u.dot(v) << "\n";
  std::cout << "u x v =\n" << u.cross(v) << "\n";
}

void coefficientWiseOperations() {
  std::cout << "////////////////// Coefficient-Wise Operations "
               "//////////////////\n";

  // Coefficient-wise math is done through the Array interface (.array()).
  Eigen::ArrayXXd m(2, 3);
  m << 1, -2, 3,
       -4, 5, -6;
  std::cout << "m =\n" << m << "\n";

  std::cout << "abs(m) =\n" << m.abs() << "\n";
  std::cout << "m^2 =\n" << m.square() << "\n";
  std::cout << "sqrt(abs(m)) =\n" << m.abs().sqrt() << "\n";
  std::cout << "exp(m) =\n" << m.exp() << "\n";
  std::cout << "log(abs(m)) =\n" << m.abs().log() << "\n";

  // min/max of two arrays, coefficient-wise.
  Eigen::ArrayXXd n = Eigen::ArrayXXd::Constant(2, 3, 0.0);
  std::cout << "min(m, 0) =\n" << m.min(n) << "\n";
  std::cout << "max(m, 0) =\n" << m.max(n) << "\n";

  // Rounding.
  Eigen::ArrayXd r(3);
  r << -1.4, 2.5, 3.7;
  std::cout << "floor: " << r.floor().transpose() << "\n";
  std::cout << "ceil:  " << r.ceil().transpose() << "\n";
  std::cout << "round: " << r.round().transpose() << "\n";

  // Predicate checks for special floating-point values.
  Eigen::ArrayXd special(3);
  special << 1.0, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN();
  std::cout << "isFinite: " << special.isFinite().transpose() << "\n";
  std::cout << "isInf:    " << special.isInf().transpose() << "\n";
  std::cout << "isNaN:    " << special.isNaN().transpose() << "\n";
}

void maskingElements() {
  std::cout << "////////////////// Masking Elements //////////////////\n";
  // Replace each element with the corresponding element of P or Q depending on
  // whether the value in R is below a threshold: (R < threshold ? P : Q).
  const int rows = 3, cols = 2;
  Eigen::MatrixXf R = Eigen::MatrixXf::Random(rows, cols);
  Eigen::MatrixXf P = Eigen::MatrixXf::Constant(rows, cols, 1.0);
  Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(rows, cols);

  float threshold = 0.5f;
  Eigen::MatrixXf masked = (R.array() < threshold).select(P, Q);

  std::cout << "R =\n" << R << "\n";
  std::cout << "masked (R < 0.5 ? 1 : 0) =\n" << masked << "\n";
}

void reductions() {
  std::cout << "////////////////// Reductions //////////////////\n";
  Eigen::MatrixXd m(2, 3);
  m << 1, 2, 3,
       4, 5, 6;
  std::cout << "m =\n" << m << "\n";

  Eigen::Index minRow, minCol, maxRow, maxCol;
  std::cout << "min element: " << m.minCoeff(&minRow, &minCol) << " at ("
            << minRow << ", " << minCol << ")\n";
  std::cout << "max element: " << m.maxCoeff(&maxRow, &maxCol) << " at ("
            << maxRow << ", " << maxCol << ")\n";

  std::cout << "column-wise max: " << m.colwise().maxCoeff() << "\n";
  std::cout << "row-wise mean:   " << m.rowwise().mean().transpose() << "\n";

  std::cout << "sum:   " << m.sum() << "\n";
  std::cout << "mean:  " << m.mean() << "\n";
  std::cout << "prod:  " << m.prod() << "\n";
  std::cout << "trace: " << m.topLeftCorner(2, 2).trace() << "\n";

  std::cout << "L2 (Frobenius) norm: " << m.norm() << "\n";
  std::cout << "L-infinity norm:     " << m.lpNorm<Eigen::Infinity>() << "\n";

  std::cout << "all elements > 0? " << (m.array() > 0).all() << "\n";
  std::cout << "any element > 5?  " << (m.array() > 5).any() << "\n";
  std::cout << "count of elements > 3: " << (m.array() > 3).count() << "\n";

  // Rank via full-pivot LU decomposition.
  Eigen::MatrixXd rankDeficient(3, 3);
  rankDeficient << 1, 2, 3,
                   2, 4, 6,   // = 2 * row 0
                   1, 0, 1;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(rankDeficient);
  std::cout << "rank of a rank-deficient 3x3 matrix: " << lu.rank() << "\n";
}

int main() {
  matrixArithmetic();
  dotAndCrossProduct();
  coefficientWiseOperations();
  maskingElements();
  reductions();
  return 0;
}
