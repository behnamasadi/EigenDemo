# Eigen Examples and Snippets
This repository contains various examples of using Eigen library. List of some of the snippets:

## Eigen unaryExpr (Function Pointer, Lambda Expression) Example
```
double ramp(double x)
{
    if (x > 0)
        return x;
    else
        return 0;
}

void unaryExprExample()
{
    //unaryExpr is a const function so it will not give you the possibility to modify values in place
    Eigen::ArrayXd x = Eigen::ArrayXd::Random(5);
    std::cout<<x <<std::endl;
    x = x.unaryExpr([](double elem) // changed type of parameter
    {
        return elem < 0.0 ? 0.0 : 1.0; // return instead of assignment
    });
    std::cout<<x <<std::endl;
    std::cout << x.unaryExpr(std::ptr_fun(ramp)) << std::endl;

}
```
## Matrix Decomposition with Eigen: QR, Cholesky Decomposition LU, UL
```
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
*/
}

double exp(double x) // the functor we want to apply
{
    std::setprecision(5);
        return std::trunc(x);
}

void gramSchmidtOrthogonalization(Eigen::MatrixXd &matrix,Eigen::MatrixXd &orthonormalMatrix)
{
/*
    In this method you make every column perpendicular to it's previous columns,
    here if a and b are representation vector of two columns, c=b-((b.a)/|a|).a
        ^
       /
    b /
     /
    /
    ---------->
        a

        ^
       /|
    b / |
     /  | c
    /   |
    ---------->
        a

    you just have to normilze every vector after make it perpendicular to previous columns
    so:
    q1=a.normalized();
    q2=b-(b.q1).q1
    q2=q2.normalized();
    q3=c-(c.q1).q1 - (c.q2).q2
    q3=q3.normalized();


    Now we have Q, but we want A=QR so we just multiply both side by Q.transpose(), since Q is orthonormal, Q*Q.transpose() is I
    A=QR;
    Q.transpose()*A=R;
*/
    Eigen::VectorXd col;
    for(int i=0;i<matrix.cols();i++)
    {
        col=matrix.col(i);
        col=col.normalized();
        for(int j=0;j<i-1;j++)
        {
            //orthonormalMatrix.col(i)
        }

        orthonormalMatrix.col(i)=col;
    }
    Eigen::MatrixXd A(4,3);

    A<<1,2,3,-1,1,1,1,1,1,1,1,1;
    Eigen::Vector4d a=A.col(0);
    Eigen::Vector4d b=A.col(1);
    Eigen::Vector4d c=A.col(2);

    Eigen::Vector4d q1=  a.normalized();
    Eigen::Vector4d q2=b-(b.dot(q1))*q1;
    q2=q2.normalized();

    Eigen::Vector4d q3=c-(c.dot(q1))*q1 - (c.dot(q2))*q2;
    q3=q3.normalized();

    std::cout<< "q1:"<<std::endl;
    std::cout<< q1<<std::endl;
    std::cout<< "q2"<<std::endl;
    std::cout<< q2<<std::endl;
    std::cout<< "q3:"<<std::endl;
    std::cout<< q3<<std::endl;

    Eigen::MatrixXd Q(4,3);
    Q.col(0)=q1;
    Q.col(1)=q2;
    Q.col(2)=q3;

    Eigen::MatrixXd R(3,3);
    R=Q.transpose()*(A);


    std::cout<<"Q"<<std::endl;
    std::cout<< Q<<std::endl;


    std::cout<<"R"<<std::endl;
    std::cout<< R.unaryExpr(std::ptr_fun(exp))<<std::endl;



    //MatrixXd A(4,3), thinQ(4,3), Q(4,4);

    Eigen::MatrixXd thinQ(4,3), q(4,4);

    //A.setRandom();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    q = qr.householderQ();
    thinQ.setIdentity();
    thinQ = qr.householderQ() * thinQ;
    std::cout << "Q computed by Eigen" << "\n\n" << thinQ << "\n\n";
    std::cout << q << "\n\n" << thinQ << "\n\n";


}

void gramSchmidtOrthogonalizationExample()
{
    Eigen::MatrixXd matrix(3,4),orthonormalMatrix(3,4) ;
    matrix=Eigen::MatrixXd::Random(3,4);////A.setRandom();


    gramSchmidtOrthogonalization(matrix,orthonormalMatrix);
}

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
          └	  ┘
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
```
## Eigen Memory Mapping
```
struct point {
    double a;
    double b;
};

void EigenMapExample()
{
    ////////////////////////////////////////First Example/////////////////////////////////////////
    Eigen::VectorXd solutionVec(12,1);
    solutionVec<<1,2,3,4,5,6,7,8,9,10,11,12;
    Eigen::Map<Eigen::MatrixXd> solutionColMajor(solutionVec.data(),4,3);

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> >solutionRowMajor (solutionVec.data());


    std::cout << "solutionColMajor: "<< std::endl;
    std::cout << solutionColMajor<< std::endl;

    std::cout << "solutionRowMajor"<< std::endl;
    std::cout << solutionRowMajor<< std::endl;

    ////////////////////////////////////////Second Example/////////////////////////////////////////

    // https://stackoverflow.com/questions/49813340/stdvectoreigenvector3d-to-eigenmatrixxd-eigen

    int array[9];
    for (int i = 0; i < 9; ++i) {
        array[i] = i;
    }

    Eigen::MatrixXi a(9, 1);
    a = Eigen::Map<Eigen::Matrix3i>(array);
    std::cout << a << std::endl;

    std::vector<point> pointsVec;
    point point1, point2, point3;

    point1.a = 1.0;
    point1.b = 1.5;

    point2.a = 2.4;
    point2.b = 3.5;

    point3.a = -1.3;
    point3.b = 2.4;

    pointsVec.push_back(point1);
    pointsVec.push_back(point2);
    pointsVec.push_back(point3);

    Eigen::Matrix2Xd pointsMatrix2d = Eigen::Map<Eigen::Matrix2Xd>(
        reinterpret_cast<double*>(pointsVec.data()), 2,  long(pointsVec.size()));

    Eigen::MatrixXd pointsMatrixXd = Eigen::Map<Eigen::MatrixXd>(
        reinterpret_cast<double*>(pointsVec.data()), 2, long(pointsVec.size()));

    std::cout << pointsMatrix2d << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << pointsMatrixXd << std::endl;
    std::cout << "==============================" << std::endl;

    std::vector<Eigen::Vector3d> eigenPointsVec;
    eigenPointsVec.push_back(Eigen::Vector3d(2, 4, 1));
    eigenPointsVec.push_back(Eigen::Vector3d(7, 3, 9));
    eigenPointsVec.push_back(Eigen::Vector3d(6, 1, -1));
    eigenPointsVec.push_back(Eigen::Vector3d(-6, 9, 8));

    Eigen::MatrixXd pointsMatrix = Eigen::Map<Eigen::MatrixXd>(eigenPointsVec[0].data(), 3, long(eigenPointsVec.size()));

    std::cout << pointsMatrix << std::endl;
    std::cout << "==============================" << std::endl;

    pointsMatrix = Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double*>(eigenPointsVec.data()), 3, long(eigenPointsVec.size()));

    std::cout << pointsMatrix << std::endl;

    std::vector<double> aa = { 1, 2, 3, 4 };
    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(aa.data(), long(aa.size()));
}
```
## Eigen Arrays, Matrices and Vectors: Definition, Initialization Resizing, Populating and Coefficient Wise Operations
```
    std::cout <<"/////////////////Definition////////////////"<<std::endl;

/*
    Definition of vectors and matrices in Eigen comes in the following form:

    Eigen::MatrixSizeType
    Eigen::VectorSizeType
    Eigen::ArraySizeType

    Size can be 2,3,4 for fixed size square matrices or X for dynamic size
    Type can be:
    i for integer,
    f for float,
    d for double,
    c for complex,
    cf for complex float,
    cd for complex double.
*/

    // Vector3f is a fixed column vector of 3 floats:
    Eigen::Vector3f objVector3f;

    // RowVector2i is a fixed row vector of 3 integer:
    Eigen::RowVector2i objRowVector2i;

    // VectorXf is a column vector of size 10 floats:
    Eigen::VectorXf objv(10);


    Eigen::Matrix4d m; // 4x4 double

    Eigen::Matrix4cd objMatrix4cd; // 4x4 double complex


    //a is a 3x3 matrix, with a static float[9] array of uninitialized coefficients,
    Eigen::Matrix3f a;

    //b is a dynamic-size matrix whose size is currently 0x0, and whose array of coefficients hasn't yet been allocated at all.
    Eigen::MatrixXf b;

    //A is a 10x15 dynamic-size matrix, with allocated but currently uninitialized coefficients.
    Eigen::MatrixXf A(10, 15);

    //V is a dynamic-size vector of size 30, with allocated but currently uninitialized coefficients.
    Eigen::VectorXf V(30);


    //Template style definition
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>  &matrix

    Eigen::Matrix<double, 2, 3> my_matrix;
    my_matrix << 1, 2, 3, 4, 5, 6;

    // ArrayXf
    Eigen::Array<float, Eigen::Dynamic, 1> a1;
    // Array3f
    Eigen::Array<float, 3, 1> a2;
    // ArrayXXd
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> a3;
    // Array33d
    Eigen::Array<double, 3, 3> a4;
    Eigen::Matrix3d matrix_from_array = a4.matrix();


    std::cout <<"///////////////////Initialization//////////////////"<< std::endl;

    Eigen::Matrix2d a_2d;
    a_2d.setRandom();

    a_2d.setConstant(4.3);

    Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(6,6);

    Eigen::MatrixXd zeros=Eigen::MatrixXd::Zero(3, 3);

    Eigen::ArrayXXf table(10, 4);
    table.col(0) = Eigen::ArrayXf::LinSpaced(10, 0, 90);


    std::cout <<"/////////////Matrix Coefficient Wise Operations///////////////////"<< std::endl;
    int i,j;
    std::cout << my_matrix << std::endl;
    std::cout << my_matrix.transpose()<< std::endl;

    std::cout<<my_matrix.minCoeff(&i, &j)<<std::endl;
    std::cout<<my_matrix.maxCoeff(&i, &j)<<std::endl;
    std::cout<<my_matrix.prod()<<std::endl;
    std::cout<<my_matrix.sum()<<std::endl;
    std::cout<<my_matrix.mean()<<std::endl;
    std::cout<<my_matrix.trace()<<std::endl;
    std::cout<<my_matrix.colwise().mean()<<std::endl;
    std::cout<<my_matrix.rowwise().maxCoeff()<<std::endl;
    std::cout<<my_matrix.lpNorm<2>()<<std::endl;
    std::cout<<my_matrix.lpNorm<Eigen::Infinity>()<<std::endl;
    std::cout<<(my_matrix.array()>0).all()<<std::endl;// if all elemnts are positive
    std::cout<<(my_matrix.array()>2).any()<<std::endl;//if any element is greater than 2
    std::cout<<(my_matrix.array()>1).count()<<std::endl;// count the number of elements greater than 1
    std::cout << my_matrix.array() - 2 << std::endl;
    std::cout << my_matrix.array().abs() << std::endl;
    std::cout << my_matrix.array().square() << std::endl;
    std::cout << my_matrix.array() * my_matrix.array() << std::endl;
    std::cout << my_matrix.array().exp() << std::endl;
    std::cout << my_matrix.array().log() << std::endl;
    std::cout << my_matrix.array().sqrt() << std::endl;

    std::cout <<"//////////////////Block Elements Access////////////////////"<< std::endl;
    //Block of size (p,q), starting at (i,j)	matrix.block(i,j,p,q)
    Eigen::MatrixXf mat(4, 4);
    mat << 1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16;
    std::cout << "Block in the middle" << std::endl;
    std::cout << mat.block<2, 2>(1, 1) << std::endl;

    for (int i = 1; i <= 3; ++i)
    {
        std::cout << "Block of size " << i << "x" << i << std::endl;
        std::cout << mat.block(0, 0, i, i) << std::endl;
    }


    /////////////build matrix from vector, resizing matrix, dynamic size ////////////////
    Eigen::MatrixXd dynamicMatrix;
    int rows, cols;
    rows=3;
    cols=2;
    dynamicMatrix.resize(rows,cols);
    dynamicMatrix<<-1,7,3,4,5,1;

    //If you want a conservative variant of resize() which does not change the coefficients, use conservativeResize()
    dynamicMatrix.conservativeResize(dynamicMatrix.rows(), dynamicMatrix.cols()+1);
    dynamicMatrix.col(dynamicMatrix.cols()-1) = Eigen::Vector3d(1, 4, 0);

    dynamicMatrix.conservativeResize(dynamicMatrix.rows(), dynamicMatrix.cols()+1);
    dynamicMatrix.col(dynamicMatrix.cols()-1) = Eigen::Vector3d(5, -8, 6);
/*
    you should expect this:
     1  7  1  5
     3  4  4 -8
     5  1  0  6

*/
    std::cout<< dynamicMatrix<<std::endl;
```
Refrences [1](http://ros-developer.com/2019/03/27/eigen-unaryexpr-function-pointer-lambda-expression-example/) 
[2](http://ros-developer.com/2019/03/27/matrix-decomposition-with-eigen-qr-cholesky-decomposition-lu-ul/) [3](http://ros-developer.com/2019/03/27/eigen-memory-mapping/) [4](http://ros-developer.com/2019/03/27/eigen-arrays-matrices-and-vectors-definition-initialization-and-coefficient-wise-operations/)

![alt text](https://img.shields.io/badge/license-BSD-blue.svg)
