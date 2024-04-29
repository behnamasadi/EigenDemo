/*
https://stackoverflow.com/questions/34662940/how-to-compute-basis-of-nullspace-with-eigen-library
*/
#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;

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



int main()
{
    //completeOrthogonalDecompositionNullSpace();
}
