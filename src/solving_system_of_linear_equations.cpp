#include <Eigen/Dense>
// #include <ctime>
#include <iostream>
#include <time.h>

void solvingSystemOfLinearEquationsUsingSVD() {

  Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
  std::cout << "Here is the matrix A:\n" << A << std::endl;
  Eigen::VectorXf b = Eigen::VectorXf::Random(3);
  std::cout << "Here is the right hand side b:\n" << b << std::endl;
  std::cout << "The least-squares solution is:\n"
            << A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b)
            << std::endl;
}

void solvingSystemOfLinearEquationsUsingQRDecomposition() {

  Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
  Eigen::VectorXf b = Eigen::VectorXf::Random(3);
  std::cout << "The solution using the QR decomposition is:\n"
            << A.colPivHouseholderQr().solve(b) << std::endl;
}

void solvingSystemOfLinearEquationsUsingCompleteOrthogonalDecomposition() {
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
  Eigen::VectorXf b = Eigen::VectorXf::Random(3);

  //    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<double,
  //    Eigen::Dynamic, Eigen::Dynamic>> cod;
  //    std::cout<<cod.solve(b)<<std::endl;
}

void solvingSystemOfLinearEquationsUsingCholeskyDecomposition() {
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
  Eigen::VectorXf b = Eigen::VectorXf::Random(3);
  std::cout << "The solution using normal equations is:\n"
            << (A.transpose() * A).ldlt().solve(A.transpose() * b) << std::endl;
}

void solver() {
  // EIGEN_STACK_ALLOCATION_LIMIT is 200
#define MATRIX_SIZE 100
  using namespace Eigen;
  using namespace std;
  // Solving equations
  // We solve the equation of matrix_NN ∗ x = v_Nd

  Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN =
      MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
  matrix_NN =
      matrix_NN * matrix_NN.transpose(); // Guarantee semi−positive definite
  Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

  clock_t time_stt = clock(); // timing
  // Direct inversion
  Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
  cout << "=======================  time of normal inverse is: "

       << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC
       << "ms ======================= " << endl;
  // cout << "x = " << x.transpose() << endl;

  // Usually solved by matrix decomposition, such as QR decomposition, the speed
  // will be much faster
  time_stt = clock();
  x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout << "======================= time of Qr decomposition is: "

       << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC
       << "ms ======================= " << endl;
  // cout << "x = " << x.transpose() << endl;

  // For positive definite matrices, you can also use cholesky decomposition to
  // solve equations.
  time_stt = clock();
  x = matrix_NN.ldlt().solve(v_Nd);
  cout << "======================= time of ldlt decomposition is "
       << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC
       << "ms ======================= " << endl;
  // cout << "x = " << x.transpose() << endl;
}

int main() {
  // solvingSystemOfLinearEquationsSVD();
  // solvingSystemOfLinearEquationsUsingCompleteOrthogonalDecomposition();
  solver();
}
