// Chapter 3: Matrix condition number and numerical stability.
//
// The condition number (in the 2-norm) is the ratio of the largest to the
// smallest singular value. A large condition number means the matrix is
// ill-conditioned: small perturbations of the input can cause large changes in
// the solution of a linear system, so results lose numerical precision.
#include <Eigen/Dense>
#include <iostream>

double conditionNumber(const Eigen::MatrixXd &m) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(m);
  const auto &sv = svd.singularValues();
  return sv(0) / sv(sv.size() - 1);
}

int main() {
  Eigen::Matrix2d wellConditioned;
  wellConditioned << 1, 0, 0, 1;

  Eigen::Matrix2d illConditioned;
  illConditioned << 1, 1, 1, 1.0001;

  std::cout << "condition number of the identity matrix: "
            << conditionNumber(wellConditioned) << "\n";
  std::cout << "condition number of a near-singular matrix: "
            << conditionNumber(illConditioned) << "\n";

  return 0;
}
