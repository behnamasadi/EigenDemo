// Chapter 10: Rigid point-cloud registration (Kabsch / Umeyama) via SVD.
//
// Given two corresponding point sets P and Q = R*P + t (+ noise), recover the
// rotation R and translation t that best align them in the least-squares sense.
// This is the core step of ICP / scan matching. The rotation comes from the SVD
// of the cross-covariance matrix H = sum (p_i - p_bar)(q_i - q_bar)^T.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

int main() {
  // Ground-truth transform we will try to recover.
  Eigen::Matrix3d R_true = (Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()))
                               .toRotationMatrix();
  Eigen::Vector3d t_true(0.4, -0.2, 1.0);

  // Source points (3 x N): each column is a 3D point.
  Eigen::Matrix<double, 3, 6> P;
  P << 0, 1, 0, 1, 2, -1, 0, 0, 1, 1, 1, 2, 0, 0, 0, 1, 2, 1;
  Eigen::Matrix<double, 3, 6> Q = (R_true * P).colwise() + t_true;

  // ----- Kabsch algorithm (manual SVD) -----
  Eigen::Vector3d cP = P.rowwise().mean();
  Eigen::Vector3d cQ = Q.rowwise().mean();
  Eigen::Matrix<double, 3, 6> Pc = P.colwise() - cP;
  Eigen::Matrix<double, 3, 6> Qc = Q.colwise() - cQ;
  Eigen::Matrix3d H = Pc * Qc.transpose(); // 3x3 cross-covariance

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();

  // Guard against an accidental reflection (det must be +1 for a rotation).
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = (V * U.transpose()).determinant();
  Eigen::Matrix3d R_est = V * D * U.transpose();
  Eigen::Vector3d t_est = cQ - R_est * cP;

  std::cout << "----- Kabsch (manual SVD) -----\n";
  std::cout << "rotation error (deg): "
            << Eigen::AngleAxisd(R_est.transpose() * R_true).angle() * 180.0 /
                   M_PI
            << "\n";
  std::cout << "translation error:    " << (t_est - t_true).norm() << "\n\n";

  // ----- Eigen::umeyama (single call; pass true to also estimate scale) -----
  Eigen::Matrix4d T = Eigen::umeyama(P, Q, /*with_scaling=*/false);
  std::cout << "----- Eigen::umeyama transform -----\n" << T << "\n";
  return 0;
}
