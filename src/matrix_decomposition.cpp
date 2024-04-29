#include <iostream>
#include <Eigen/Dense>
#include <vector>

void choleskyDecompositionExample()
{
/*
Positive-definite matrix:
    Matrix Mnxn is said to be positive definite if the scalar zTMz is strictly positive for every non-zero
    column vector z n real numbers. zTMz>0


Hermitian matrix:
    Matrix Mnxn is said to be positive definite if the scalar z*Mz is strictly positive for every non-zero
    column vector z n real numbers. z*Mz>0
    z* is the conjugate transpose of z.

Positive semi-definite same as above except zTMz>=0 or  z*Mz>=0


    Example:
            ┌2  -1  0┐
        M=  |-1  2 -1|
            |0 - 1  2|
            └        ┘
          ┌ a ┐
        z=| b |
          | c |
          └   ┘
        zTMz=a^2 +c^2+ (a-b)^2+ (b-c)^2

Cholesky decomposition:
Cholesky decomposition of a Hermitian positive-definite matrix A is:
    A=LL*

    L is a lower triangular matrix with real and positive diagonal entries
    L* is the conjugate transpose of L
*/

    Eigen::MatrixXd A(3,3);
    A << 6, 0, 0, 0, 4, 0, 0, 0, 7;
    Eigen::MatrixXd L( A.llt().matrixL() );
    Eigen::MatrixXd L_T=L.adjoint();//conjugate transpose

    std::cout << "L" << std::endl;
    std::cout << L << std::endl;
    std::cout << "L_T" << std::endl;
    std::cout << L_T << std::endl;
    std::cout << "A" << std::endl;
    std::cout << A << std::endl;
    std::cout << "L*L_T" << std::endl;
    std::cout << L*L_T << std::endl;

}


void qRDecomposition(Eigen::MatrixXd &A,Eigen::MatrixXd &Q, Eigen::MatrixXd &R)
{
    /*
        A=QR
        Q: is orthogonal matrix-> columns of Q are orthonormal
        R: is upper triangulate matrix
        this is possible when columns of A are linearly indipendent
    */

    Eigen::MatrixXd thinQ(A.rows(),A.cols() ), q(A.rows(),A.rows());

    Eigen::HouseholderQR<Eigen::MatrixXd> householderQR(A);
    q = householderQR.householderQ();
    thinQ.setIdentity();
    Q = householderQR.householderQ() * thinQ;
    R=Q.transpose()*A;
}

void qRExample()
{

    Eigen::MatrixXd A;
    A.setRandom(3,4);

    std::cout<<"A" <<std::endl;
    std::cout<<A <<std::endl;
    Eigen::MatrixXd Q(A.rows(),A.rows());
    Eigen::MatrixXd R(A.rows(),A.cols());

    /////////////////////////////////HouseholderQR////////////////////////
    Eigen::MatrixXd thinQ(A.rows(),A.cols() ), q(A.rows(),A.rows());

    Eigen::HouseholderQR<Eigen::MatrixXd> householderQR(A);
    q = householderQR.householderQ();
    thinQ.setIdentity();
    Q = householderQR.householderQ() * thinQ;

    std::cout << "HouseholderQR" <<std::endl;

    std::cout << "Q" <<std::endl;
    std::cout << Q <<std::endl;

    R = householderQR.matrixQR().template  triangularView<Eigen::Upper>();
    std::cout << R<<std::endl;
    std::cout << R.rows()<<std::endl;
    std::cout << R.cols()<<std::endl;


    R=Q.transpose()*A;
// 	std::cout << "R" <<std::endl;
// 	std::cout << R<<std::endl;

    std::cout << "A-Q*R" <<std::endl;
    std::cout << A-Q*R <<std::endl;

    /////////////////////////////////ColPivHouseholderQR////////////////////////
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> colPivHouseholderQR(A.rows(), A.cols());
    colPivHouseholderQR.compute(A);
    //R = colPivHouseholderQR.matrixR().template triangularView<Upper>();
    R = colPivHouseholderQR.matrixR();
    Q = colPivHouseholderQR.matrixQ();

    std::cout << "ColPivHouseholderQR" <<std::endl;

    std::cout << "Q" <<std::endl;
    std::cout << Q <<std::endl;

    std::cout << "R" <<std::endl;
    std::cout << R <<std::endl;

    std::cout << "A-Q*R" <<std::endl;
    std::cout << A-Q*R <<std::endl;

    /////////////////////////////////FullPivHouseholderQR////////////////////////
    std::cout << "FullPivHouseholderQR" <<std::endl;

    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> fullPivHouseholderQR(A.rows(), A.cols());
    fullPivHouseholderQR.compute(A);
    Q=fullPivHouseholderQR.matrixQ();
    R=fullPivHouseholderQR.matrixQR().template  triangularView<Eigen::Upper>();

    std::cout << "Q" <<std::endl;
    std::cout << Q <<std::endl;

    std::cout << "R" <<std::endl;
    std::cout << R <<std::endl;

    std::cout << "A-Q*R" <<std::endl;
    std::cout << A-Q*R <<std::endl;

}

void lDUDecomposition()
{
/*

    L: lower triangular matrix L
    U: upper triangular matrix U
    D: is a diagonal matrix
    A=LDU


*/
}

void lUDecomposition()
{
/*
    L: lower triangular matrix L
    U: upper triangular matrix U
    A=LU


    https://www.youtube.com/watch?v=aFbjNVZNYYk&ab_channel=TobyDriscoll
    https://www.youtube.com/watch?v=mmoliBMaaQs&ab_channel=TobyDriscoll

*/
}

/*
function [U]=gramschmidt(V)
[n,k] = size(V);
U = zeros(n,k);
U(:,1) = V(:,1)/norm(V(:,1));
for i = 2:k
    U(:,i)=V(:,i);
    for j=1:i-1
        U(:,i)=U(:,i)-(U(:,j)'*U(:,i) )/(norm(U(:,j)))^2 * U(:,j);
    end
    U(:,i) = U(:,i)/norm(U(:,i));
end
end

*/

void householderTransformation()
{
/*
https://www.statlect.com/matrix-algebra/Householder-matrix
https://www.youtube.com/watch?v=b8RRyHI95V0&ab_channel=Poujh

*/
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix;
    matrix.resize(3,3);
    matrix<<1,2,1,
            2,3,2,
            1,2,3;

}
//http://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html


void denseDecompositions()
{
    //LLT
    //LDLT
    //PartialPivLU
    //FullPivLU
    //HouseholderQR
    //ColPivHouseholderQR
    //CompleteOrthogonalDecomposition
    //FullPivHouseholderQR
    //JacobiSVD
    //BDCSVD
}



int main()
{

}

