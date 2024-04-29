#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;

void computeBasisOfNullSpace()
{

    Eigen::MatrixXd A(3,4);
    A<<1 ,1 ,2, 1 ,
            3,1,4,4,
            4,-4,0,8;


    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    Eigen::MatrixXd A_null_space = lu.kernel();

    std::cout<<A_null_space  <<std::endl;




    CompleteOrthogonalDecomposition<Matrix<double, Dynamic, Dynamic> > cod;
    cod.compute(A);
    std::cout << "rank : " << cod.rank() << "\n";
    // Find URV^T
    MatrixXd V = cod.matrixZ().transpose();
    MatrixXd Null_space = V.block(0, cod.rank(),V.rows(), V.cols() - cod.rank());
    MatrixXd P = cod.colsPermutation();
    Null_space = P * Null_space; // Unpermute the columns
    // The Null space:
    std::cout << "The null space: \n" << Null_space << "\n" ;
    // Check that it is the null-space:
    std::cout << "A * Null_space = \n" << A * Null_space  << '\n';



}


int main()
{
    computeBasisOfNullSpace();
}
