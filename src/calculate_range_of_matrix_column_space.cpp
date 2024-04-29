#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;


template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
CompleteOrthogonalDecomposition(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M) {
  Eigen::CompleteOrthogonalDecomposition<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      cod(M);
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Q =
      cod.householderQ();
  return Q.leftCols(cod.rank());
}

int main() {}
