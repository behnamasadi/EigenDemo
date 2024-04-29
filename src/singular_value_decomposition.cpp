#include <iostream>
#include <Eigen/Dense>

template<typename T>
T pseudoInverse(const T &a, double epsilon = std::numeric_limits<double>::epsilon())
{
    //Eigen::DecompositionOptions flags;
    int flags;
    // For a non-square matrix
    if(a.cols()!=a.rows())
    {
        flags=Eigen::ComputeThinU | Eigen::ComputeThinV;
    }
    else
    {
        flags=Eigen::ComputeFullU | Eigen::ComputeFullV;
    }
    Eigen::JacobiSVD< T > svd(a ,flags);

    double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
    return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}



template <class MatT>
Eigen::Matrix<typename MatT::Scalar, MatT::ColsAtCompileTime, MatT::RowsAtCompileTime>
pseudoinverse(const MatT &mat, typename MatT::Scalar tolerance = typename MatT::Scalar{1e-4}) // choose appropriately
{
    typedef typename MatT::Scalar Scalar;
    auto svd = mat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &singularValues = svd.singularValues();
    Eigen::Matrix<Scalar, MatT::ColsAtCompileTime, MatT::RowsAtCompileTime> singularValuesInv(mat.cols(), mat.rows());
    singularValuesInv.setZero();
    for (unsigned int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > tolerance)
        {
            singularValuesInv(i, i) = Scalar{1} / singularValues(i);
        }
        else
        {
            singularValuesInv(i, i) = Scalar{0};
        }
    }
    return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
}



void SVD_Example()
{
/*

AX=0;
A, U, V=SVD(A);
A* U(Index of last column)=0;

1) Full SVD
    A  mxn
    U  mxm
    Σ  mxn
    V* nxn


2) Thin SVD
    A  mxn
    U  mxn
    Σ  nxn
    V* nxn

3) Compact SVD

4) Truncated SVD
Ref: https://en.wikipedia.org/wiki/Singular_value_decomposition#Thin_SVD

*/

    std::cout<<"********************** 1) Full SVD ***********************************" <<std::endl;

    Eigen::MatrixXd A;
    A.setRandom(3,4);
    std::cout<<"Matrix A" <<std::endl;
    std::cout<<A <<std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);


    std::cout<< "Size of original matrix:"<<  A.rows()<<","<<A.cols() <<std::endl;

    std::cout<< "Size of U matrix:"<<  svd.matrixU().rows()<<","<<svd.matrixU().cols() <<std::endl;

    std::cout<< "Size of Σ matrix:"<<  svd.singularValues().rows()<<","<<svd.singularValues().cols() <<std::endl;

    std::cout<< "Size of V matrix:"<<  svd.matrixV().rows()<<","<<svd.matrixV().cols() <<std::endl;

    std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;

    Eigen::MatrixXd U=svd.matrixU();
    Eigen::MatrixXd V=svd.matrixV();
    Eigen::MatrixXd Sigma(U.rows(),V.cols());
    Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(U.rows(),V.cols());

    Sigma=identity.array().colwise()* svd.singularValues().array();

    std::cout<<"Matrix U" <<std::endl;
    std::cout<<U <<std::endl;

    std::cout<<"Matrix V" <<std::endl;
    std::cout<<V <<std::endl;

    std::cout<<"Matrix Sigma" <<std::endl;
    std::cout<<Sigma <<std::endl;

    std::cout<<"This should be very close to A" <<std::endl;
    Eigen::MatrixXd A_reconstructed= U*Sigma*V.transpose();
    std::cout<<U*Sigma*V.transpose() <<std::endl;


    std::cout<<"This should be zero vector (solution of the problem A*V.col( V.cols()-1))" <<std::endl;
    std::cout<<A*V.col( V.cols()-1)<<std::endl;


    Eigen::MatrixXd diff = A - A_reconstructed;
    std::cout << "diff:\n" << diff.array().abs().sum() << "\n";


    std::cout<<"********************** 2) Thin SVD ***********************************" <<std::endl;
    Eigen::MatrixXd C;
    C.setRandom(27,18);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_thin( C, Eigen::ComputeThinU | Eigen::ComputeThinV);

    std::cout<< "Size of original matrix:"<<  C.rows()<<","<<C.cols() <<std::endl;

    std::cout<< "Size of U matrix:"<<  svd_thin.matrixU().rows()<<","<<svd_thin.matrixU().cols() <<std::endl;

    std::cout<< "Size of Σ matrix:"<<  svd_thin.singularValues().rows()<<","<<svd_thin.singularValues().cols() <<std::endl;

    std::cout<< "Size of V matrix:"<<  svd_thin.matrixV().rows()<<","<<svd_thin.matrixV().cols() <<std::endl;

    Eigen::MatrixXd C_reconstructed = svd_thin.matrixU() * svd_thin.singularValues().asDiagonal() * svd_thin.matrixV().transpose();

    std::cout << "diff:\n" << (C - C_reconstructed).array().abs().sum() << "\n";

    Eigen::MatrixXd pinv_C =svd_thin.matrixV()*svd_thin.singularValues().asDiagonal() * svd_thin.matrixU().transpose();

//    Eigen::MatrixXd pinv_C =svd.matrixV()*svd.singularValues().asDiagonal() ;

//    MatrixXd diff = Cp - C;
//    cout << "diff:\n" << diff.array().abs().sum() << "\n";

   std::cout<< "Size of pinv_C matrix:"<<  pinv_C.rows()<<","<<pinv_C.cols() <<std::endl;


   //std::cout << "pinv_C*C:\n" << pinv_C*C << "\n";


   Eigen::MatrixXd pinv = C.completeOrthogonalDecomposition().pseudoInverse();

   Eigen::MatrixXd pinv2 = pseudoInverse(C);

   std::cout << "xxx" << (pinv2*C).rows()<< "," <<(pinv2*C).cols()  << "\n";
   std::cout << "xxx" << (pinv2*C).array().abs().sum() << "\n";

   std::cout << "xxx" << (pinv-pinv2).array().abs().sum() << "\n";

/*
Ref:
    https://gist.github.com/javidcf/25066cf85e71105d57b6
    https://eigen.tuxfamily.org/bz/show_bug.cgi?id=257#c8
    https://gist.github.com/pshriwise/67c2ae78e5db3831da38390a8b2a209f
    https://math.stackexchange.com/questions/19948/pseudoinverse-matrix-and-svd
*/
}


int main()
{

}
