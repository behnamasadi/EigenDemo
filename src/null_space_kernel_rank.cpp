/*
https://stackoverflow.com/questions/34662940/how-to-compute-basis-of-nullspace-with-eigen-library
*/
#include <Eigen/Dense>
#include <iostream>

void fullPivLU()
{
//https://stackoverflow.com/questions/31041921/how-to-get-rank-of-a-matrix-in-eigen-library

    Eigen::Matrix3d mat;
    mat<<2, 1, -1,
         -3, -1, 2,
         -2, 1, 2;

    Eigen::FullPivLU<Eigen::Matrix3d> lu_decomp(mat);
    //Eigen::FullPivHouseholderQR
    auto rank = lu_decomp.rank();
    std::cout<<"Rank:" <<rank <<std::endl;


    std::cout<<"Kernel:\n" <<lu_decomp.kernel()<<std::endl;


    std::cout<<"MatrixLU:\n" <<lu_decomp.matrixLU()<<std::endl;


    std::cout<<"Determinant:" <<lu_decomp.determinant()<<std::endl;

//    lu_decomp.permutationP();
//    std::cout<<lu_decomp.kernel()<<std::endl;

//    lu_decomp.permutationQ();
//    std::cout<<lu_decomp.kernel()<<std::endl;



}

void completeOrthogonalDecompositionNullSpace()
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat37(3,7);
    mat37 = Eigen::MatrixXd::Random(3, 7);

    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > cod;
    cod.compute(mat37);
    std::cout << "rank : " << cod.rank() << "\n";
    // Find URV^T
    Eigen::MatrixXd V = cod.matrixZ().transpose();
    Eigen::MatrixXd Null_space = V.block(0, cod.rank(),V.rows(), V.cols() - cod.rank());
    Eigen::MatrixXd P = cod.colsPermutation();
    Null_space = P * Null_space; // Unpermute the columns
    // The Null space:
    std::cout << "The null space: \n" << Null_space << "\n" ;
    // Check that it is the null-space:
    std::cout << "mat37 * Null_space = \n" << mat37 * Null_space  << '\n';
}

int main()
{
    completeOrthogonalDecompositionNullSpace();
}
