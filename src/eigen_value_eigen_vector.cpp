#include <iostream>
#include <Eigen/Dense>


void eigenValueSolver()
{
    // ComplexEigenSolver<MatrixXcf> ces;
    Eigen::EigenSolver<Eigen::MatrixXd> ces;
    Eigen::MatrixXd A(4, 4);
    A(0, 0) = 18;
    A(0, 1) = -9;
    A(0, 2) = -27;
    A(0, 2) = 24;
    A(1, 0) = -9;
    A(1, 1) = 4.5;
    A(1, 2) = 13.5;
    A(1, 3) = -12;
    A(2, 0) = -27;
    A(2, 1) = 13.5;
    A(2, 2) = 40.5;
    A(2, 3) = -36;
    A(3, 0) = 24;
    A(3, 1) = -12;
    A(3, 2) = -36;
    A(3, 3) = 32;
    ces.compute(A);
    std::cout << "The eigenvalues of A are:" << std::endl
         << ces.eigenvalues() << std::endl;
    std::cout << "The matrix of eigenvectors, V, is:" <<std::endl
         << ces.eigenvectors() << std::endl
         << std::endl;
    //  complex<float> lambda =  ces.eigenvalues()[0];
    //  cout << "Consider the first eigenvalue, lambda = " << lambda << endl;
    //  VectorXcf v = ces.eigenvectors().col(0);
    //  cout << "If v is the corresponding eigenvector, then lambda * v = " <<
    //  endl << lambda * v << endl;
    //  cout << "... and A * v = " << endl << A * v << endl << endl;
    //
    //  cout << "Finally, V * D * V^(-1) = " << endl
    //       << ces.eigenvectors() * ces.eigenvalues().asDiagonal() *
    //       ces.eigenvectors().inverse() << endl;
}

int main()
{

}
