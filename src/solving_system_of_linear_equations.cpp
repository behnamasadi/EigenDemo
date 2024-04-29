#include <Eigen/Dense>
#include <iostream>

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

void solvingSystemOfLinearEquationsUsingCompleteOrthogonalDecomposition()
{
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
    Eigen::VectorXf b = Eigen::VectorXf::Random(3);



//    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> cod;
//    std::cout<<cod.solve(b)<<std::endl;
}


void solvingSystemOfLinearEquationsUsingCholeskyDecomposition()
{
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
    Eigen::VectorXf b = Eigen::VectorXf::Random(3);
    std::cout << "The solution using normal equations is:\n"
         << (A.transpose() * A).ldlt().solve(A.transpose() * b) << std::endl;
}



int main()
{
    //solvingSystemOfLinearEquationsSVD();
    //solvingSystemOfLinearEquationsUsingCompleteOrthogonalDecomposition();
}
