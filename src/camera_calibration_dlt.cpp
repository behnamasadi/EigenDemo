// Chapter 10: Camera calibration with matrix decompositions.
//
// Two decompositions appear here:
//   * SVD  -- to solve the homogeneous system A p = 0 (Direct Linear Transform)
//             that estimates the 3x4 projection matrix P from 3D<->2D matches.
//   * QR   -- to decompose the left 3x3 block of P into the intrinsics K
//             (upper triangular) and the rotation R (orthogonal).
//
// P = K [R | t]. Given >= 6 correspondences (X_world, (u,v)) we recover P, then
// factor it back into K, R, t.
#include <Eigen/Dense>
#include <iostream>
#include <vector>

int main() {
  // ---- Ground-truth camera we will try to recover ----
  Eigen::Matrix3d K_true;
  K_true << 800, 0, 320, 0, 800, 240, 0, 0, 1;

  // A rotation built from three axis rotations.
  Eigen::Matrix3d R_true = (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()))
                               .toRotationMatrix();
  Eigen::Vector3d t_true(0.2, -0.1, 6.0); // camera in front of the points

  Eigen::MatrixXd Rt(3, 4);
  Rt.leftCols(3) = R_true;
  Rt.col(3) = t_true;
  Eigen::MatrixXd P_true = K_true * Rt; // 3x4 projection matrix

  // ---- Generate 3D points and their image projections ----
  std::vector<Eigen::Vector3d> worldPts = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0},
                                           {-1, 1, 0},  {0, 0, 1},  {-1, 0, 1},
                                           {1, 0, -1},  {0, 1, 0.5}};

  std::vector<Eigen::Vector2d> imagePts;
  for (const auto &Xw : worldPts) {
    Eigen::Vector4d Xh(Xw(0), Xw(1), Xw(2), 1.0);
    Eigen::Vector3d x = P_true * Xh; // homogeneous image point
    imagePts.emplace_back(x(0) / x(2), x(1) / x(2));
  }

  // ---- DLT: build A (2N x 12) and solve A p = 0 with the SVD ----
  const int N = static_cast<int>(worldPts.size());
  Eigen::MatrixXd A(2 * N, 12);
  for (int i = 0; i < N; ++i) {
    Eigen::Vector4d X(worldPts[i](0), worldPts[i](1), worldPts[i](2), 1.0);
    double u = imagePts[i](0), v = imagePts[i](1);
    A.row(2 * i) << -X.transpose(), Eigen::RowVector4d::Zero(),
        u * X.transpose();
    A.row(2 * i + 1) << Eigen::RowVector4d::Zero(), -X.transpose(),
        v * X.transpose();
  }

  // The solution is the right-singular vector of the smallest singular value,
  // i.e. the last column of V.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  Eigen::VectorXd p = svd.matrixV().col(11);

  Eigen::MatrixXd P_est(3, 4);
  P_est.row(0) = p.segment<4>(0).transpose();
  P_est.row(1) = p.segment<4>(4).transpose();
  P_est.row(2) = p.segment<4>(8).transpose();

  // ---- Decompose the left 3x3 block M = K R via QR ----
  // M^{-1} = R^{-1} K^{-1} = R^T K^{-1}: a QR factorization (orthogonal x
  // upper-triangular), so QR(M^{-1}) = (Q = R^T)(R_a = K^{-1}).
  Eigen::Matrix3d M = P_est.leftCols(3);
  Eigen::HouseholderQR<Eigen::Matrix3d> qr(M.inverse());
  Eigen::Matrix3d Q = qr.householderQ();
  Eigen::Matrix3d Ra = qr.matrixQR().triangularView<Eigen::Upper>();

  Eigen::Matrix3d R_est = Q.transpose();
  Eigen::Matrix3d K_est = Ra.inverse();

  // Force a positive diagonal on K (and keep K*R consistent).
  for (int i = 0; i < 3; ++i) {
    if (K_est(i, i) < 0) {
      K_est.col(i) *= -1;
      R_est.row(i) *= -1;
    }
  }
  K_est /= K_est(2, 2); // normalise so K(2,2) = 1
  if (R_est.determinant() < 0)
    R_est *= -1;

  std::cout << "True K:\n" << K_true << "\n\n";
  std::cout << "Recovered K (via SVD-DLT + QR):\n" << K_est << "\n\n";
  std::cout << "max |K_true - K_est| = "
            << (K_true - K_est).cwiseAbs().maxCoeff() << "\n";
  return 0;
}
