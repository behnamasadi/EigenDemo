// Chapter 10: A tiny 1D pose-graph SLAM, solved as linear least squares.
//
// This shows the linear-algebra core of graph SLAM:
//   * Each measurement is a row of the Jacobian J (a "factor").
//   * Gauss-Newton solves the normal equations (J^T J) dx = -J^T r.
//   * H = J^T J is sparse, symmetric positive-definite -> Cholesky (LLT).
//   * Forming J^T J squares the condition number; QR on J avoids that.
//
// Poses x0..x4 lie on a line. We have a prior anchoring x0, odometry between
// consecutive poses, and one loop closure x4 - x0. The problem is linear, so
// Gauss-Newton converges in a single step.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

int main() {
  const int nPoses = 5;

  // Each measurement: (row in J) selects pose(s); residual r = prediction - z.
  // We assemble J (m x n) and the residual vector r (m), all linear here.
  struct Factor {
    int i, j;
    double z;
  }; // measures x_j - x_i = z (i<0 => prior on x_j)
  std::vector<Factor> factors = {
      {-1, 0, 0.0}, // prior: x0 = 0
      {0, 1, 1.0},  // odometry x1 - x0 = 1
      {1, 2, 1.0},  // odometry x2 - x1 = 1
      {2, 3, 1.0},  // odometry x3 - x2 = 1
      {3, 4, 1.0},  // odometry x4 - x3 = 1
      {0, 4, 3.9},  // loop closure x4 - x0 = 3.9 (slightly inconsistent)
  };

  const int m = static_cast<int>(factors.size());
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(m, nPoses);
  Eigen::VectorXd z(m);
  for (int k = 0; k < m; ++k) {
    const Factor &f = factors[k];
    if (f.i < 0) { // prior on x_j
      J(k, f.j) = 1.0;
    } else { // x_j - x_i
      J(k, f.j) = 1.0;
      J(k, f.i) = -1.0;
    }
    z(k) = f.z;
  }

  // Linear least squares: minimize ||J x - z||^2.
  // Normal equations: (J^T J) x = J^T z, solved with sparse Cholesky.
  Eigen::SparseMatrix<double> Js = J.sparseView();
  Eigen::SparseMatrix<double> H = (Js.transpose() * Js).pruned();
  Eigen::VectorXd b = Js.transpose() * z;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol(H);
  Eigen::VectorXd x_chol = chol.solve(b);

  // The same solve via QR on J directly (more stable: avoids squaring kappa).
  Eigen::VectorXd x_qr = J.colPivHouseholderQr().solve(z);

  std::cout << "Information matrix H = J^T J (sparse, tridiagonal + loop):\n"
            << Eigen::MatrixXd(H) << "\n\n";
  std::cout << "Solution via sparse Cholesky (LDLT): " << x_chol.transpose()
            << "\n";
  std::cout << "Solution via QR on J:                " << x_qr.transpose()
            << "\n\n";

  // Conditioning: forming J^T J squares the condition number.
  Eigen::MatrixXd Hd(H);
  Eigen::JacobiSVD<Eigen::MatrixXd> svdJ(J);
  Eigen::JacobiSVD<Eigen::MatrixXd> svdH(Hd);
  double condJ = svdJ.singularValues()(0) /
                 svdJ.singularValues()(svdJ.singularValues().size() - 1);
  double condH = svdH.singularValues()(0) /
                 svdH.singularValues()(svdH.singularValues().size() - 1);
  std::cout << "cond(J)      = " << condJ << "\n";
  std::cout << "cond(J^T J)  = " << condH
            << "   (~ cond(J)^2 = " << condJ * condJ << ")\n";
  return 0;
}
