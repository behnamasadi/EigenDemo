// Chapter 9: Gauss-Newton inverse kinematics for a 3-link planar robot.
//
// The arm has three revolute joints with angles q = (q0, q1, q2) and link
// lengths (L1, L2, L3). The end-effector pose is p = (x, y, phi), where phi is
// the orientation. Given a goal pose we solve for q with Gauss-Newton:
//     q <- q + J^+ (p_goal - p(q))
// where J is the Jacobian of the forward kinematics and J^+ its pseudoinverse.
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

constexpr double L1 = 1.0, L2 = 1.0, L3 = 1.0;

// Forward kinematics: joint angles -> end-effector pose (x, y, phi).
Eigen::Vector3d forwardKinematics(const Eigen::Vector3d &q) {
  double a0 = q(0);
  double a1 = q(0) + q(1);
  double a2 = q(0) + q(1) + q(2);
  Eigen::Vector3d p;
  p(0) = L1 * std::cos(a0) + L2 * std::cos(a1) + L3 * std::cos(a2);
  p(1) = L1 * std::sin(a0) + L2 * std::sin(a1) + L3 * std::sin(a2);
  p(2) = a2;
  return p;
}

// Numerical (central-difference) Jacobian of the forward kinematics, 3x3.
Eigen::Matrix3d numericalDifferentiationFK(const Eigen::Vector3d &q) {
  Eigen::Matrix3d J;
  const double h = 1e-6;
  for (int j = 0; j < 3; ++j) {
    Eigen::Vector3d qp = q, qm = q;
    qp(j) += h;
    qm(j) -= h;
    J.col(j) = (forwardKinematics(qp) - forwardKinematics(qm)) / (2 * h);
  }
  return J;
}

// Moore-Penrose pseudoinverse via SVD (robust near singular configurations).
Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd &m) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU |
                                               Eigen::ComputeThinV);
  double tol = 1e-9 * std::max(m.rows(), m.cols()) * svd.singularValues()(0);
  Eigen::VectorXd invSv = svd.singularValues();
  for (int i = 0; i < invSv.size(); ++i)
    invSv(i) = (invSv(i) > tol) ? 1.0 / invSv(i) : 0.0;
  return svd.matrixV() * invSv.asDiagonal() * svd.matrixU().transpose();
}

int main() {
  Eigen::Vector3d q(0.1, 0.1, 0.1);           // initial guess
  Eigen::Vector3d goal(1.0, 1.0, M_PI / 2.0); // desired pose (x, y, phi)

  const double epsilon = 1e-10;
  const int maxIters = 200;

  int i = 0;
  for (; i < maxIters; ++i) {
    Eigen::Vector3d error = goal - forwardKinematics(q);
    if (error.squaredNorm() < epsilon)
      break;
    Eigen::Matrix3d jacobian = numericalDifferentiationFK(q);
    q += pseudoInverse(jacobian) * error;
  }

  Eigen::Vector3d reached = forwardKinematics(q);
  std::cout << "iterations: " << i << "\n";
  std::cout << "q       = " << q.transpose() << "\n";
  std::cout << "goal    = " << goal.transpose() << "\n";
  std::cout << "reached = " << reached.transpose() << "\n";
  std::cout << "residual = " << (goal - reached).norm() << "\n";
  return 0;
}
