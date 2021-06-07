#include <iostream>
#include <vector>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

//The value of KDL_FOUND has been set via target_compile_definitions in CMake

#if KDL_FOUND==1
#include <kdl/frames.hpp>
#endif

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

struct point {
    double a;
    double b;
};

void eigenMapExample()
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

void demo()
{
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
}

void checkMatrixsimilarity()
{
    // EXPECT_NEAR should be used element wise
    // This could be also used
    // ASSERT_TRUE(((translation - expectedTranslation).norm() < precision);

    // Pointwise() matcher could be also used
    // EXPECT_THAT(result_array, Pointwise(NearWithPrecision(0.1), expected_array));
}

Eigen::Matrix3d eulerAnglesToRotationMatrix(double roll, double pitch,double yaw)
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d rotationMatrix = q.matrix();
    return rotationMatrix;
}

void transformation()
{
/*
Great Tutorial:
http://planning.cs.uiuc.edu/node102.html
http://euclideanspace.com/maths/geometry/rotations/conversions/index.htm

Tait–Bryan angles: Z1Y2X3 in the wiki page:
https://en.wikipedia.org/wiki/Euler_angles
  yaw:
      A yaw is a counterclockwise rotation of alpha about the  z-axis. The
  rotation matrix is given by

      R_z

      |cos(alpha) -sin(alpha) 0|
      |sin(apha)   cos(alpha) 0|
      |    0            0     1|

  pitch:
      R_y
      A pitch is a counterclockwise rotation of  beta about the  y-axis. The
  rotation matrix is given by

      |cos(beta)  0   sin(beta)|
      |0          1       0    |
      |-sin(beta) 0   cos(beta)|

  roll:
      A roll is a counterclockwise rotation of  gamma about the  x-axis. The
  rotation matrix is given by
      R_x
      |1          0           0|
      |0 cos(gamma) -sin(gamma)|
      |0 sin(gamma)  cos(gamma)|



      It is important to note that   R_z R_y R_x performs the roll first, then the pitch, and finally the yaw
      Roration matrix: R_z*R_y*R_x

*/
/////////////////////////////////////Rotation Matrix (Tait–Bryan)///////////////////////////////
    double roll, pitch, yaw;
    roll=M_PI/2;
    pitch=M_PI/2;
    yaw=0;//M_PI/6;
    std::cout << "Roll : " <<  roll << std::endl;
    std::cout << "Pitch : " << pitch  << std::endl;
    std::cout << "Yaw : " << yaw  << std::endl;

/////////////////////////////////////Rotation Matrix (Tait–Bryan)///////////////////////////////

    // Roll, Pitch, Yaw to Rotation Matrix
    //Eigen::AngleAxis<double> rollAngle(roll, Eigen::Vector3d(1,0,0));

    std::cout << "Roll : " <<  roll << std::endl;
    std::cout << "Pitch : " << pitch  << std::endl;
    std::cout << "Yaw : " << yaw  << std::endl;



    // Roll, Pitch, Yaw to Rotation Matrix
    //Eigen::AngleAxis<double> rollAngle(roll, Eigen::Vector3d(1,0,0));
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());



//////////////////////////////////////// Quaternion ///////////////////////////////////////////
    Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;

    
    //Quaternion to Rotation Matrix
    Eigen::Matrix3d rotationMatrix = q.matrix();
    std::cout << "3x3 Rotation Matrix" << std::endl;

    std::cout << rotationMatrix << std::endl;

    Eigen::Quaterniond quaternion_mat(rotationMatrix);
    std::cout << "Quaternion X: " << quaternion_mat.x() << std::endl;
    std::cout << "Quaternion Y: " << quaternion_mat.y() << std::endl;
    std::cout << "Quaternion Z: " << quaternion_mat.z() << std::endl;
    std::cout << "Quaternion W: " << quaternion_mat.w() << std::endl;



//////////////////////////////////////// Rodrigues ///////////////////////////////////////////


    //Rotation Matrix to Rodrigues
    Eigen::AngleAxisd rodrigues(rotationMatrix );
    std::cout<<"Rodrigues Angle:\n"<<rodrigues.angle() <<std::endl;

    std::cout<<"Rodrigues Axis:" <<std::endl;

    std::cout<<rodrigues.axis().x() <<std::endl;
    std::cout<<rodrigues.axis().y() <<std::endl;
    std::cout<<rodrigues.axis().z() <<std::endl;





    Eigen::Vector3d vector3d(2.3,3.1,1.7);
    Eigen::Vector3d vector3dNormalized=vector3d.normalized();
    double theta=M_PI/7;
    Eigen::AngleAxisd angleAxisConversion(theta,vector3dNormalized);
    Eigen::Matrix3d rotationMatrixConversion;

    // Angle Axis (Rodrigues) to Rotation Matrix
    rotationMatrixConversion=angleAxisConversion.toRotationMatrix();

    
    //Rotation Matrix to Quaternion
    
    Eigen::Quaterniond QuaternionConversion(rotationMatrixConversion);

    //Rotation Matrix to Euler Angle (Proper)
    Eigen::Vector3d euler_angles = rotationMatrixConversion.eulerAngles(2, 0, 2);

    //Eigen::Quaterniond
    Eigen::Quaterniond tmp = Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitZ())
     * Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitX())
     * Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitZ());

////////////////////////////////////////Comparing with KDL////////////////////////////////////////
#if KDL_FOUND==1
    KDL::Frame F;
    F.M = F.M.RPY(roll, pitch, yaw);
    std::cout << F.M(0, 0) << " " << F.M(0, 1) << " " << F.M(0, 2) << std::endl;
    std::cout << F.M(1, 0) << " " << F.M(1, 1) << " " << F.M(1, 2) << std::endl;
    std::cout << F.M(2, 0) << " " << F.M(2, 1) << " " << F.M(2, 2) << std::endl;

    double x, y, z, w;
    F.M.GetQuaternion(x, y, z, w);
    std::cout << "KDL Frame Quaternion:" << std::endl;
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
    std::cout << "z: " << z << std::endl;
    std::cout << "w: " << w << std::endl;
#endif


////////////////////////////////////////Comparing with KDL////////////////////////////////////////

    Eigen::Matrix3d rotation;
    rotation= eulerAnglesToRotationMatrix(roll, pitch,yaw);

    double 	txLeft, tyLeft, tzLeft;
    txLeft=-1;
    tyLeft=0.0;
    tzLeft=-4.0;

    Eigen::Affine3f t1;
    Eigen::Matrix4f M;
    Eigen::Vector3d translation;
    translation<<txLeft,tyLeft,tzLeft;

    M<<  rotation(0,0),rotation(0,1),rotation(0,2),translation(0,0)
     ,rotation(1,0),rotation(1,1),rotation(1,2),translation(1,0)
     ,rotation(2,0),rotation(2,1),rotation(2,2),translation(2,0)
     ,0,0,0,1;


    t1 = M;


    Eigen::Affine3d transform_2 = Eigen::Affine3d::Identity();


    // Define a translation of 2.5 meters on the x axis.
    transform_2.translation() << 2.5, 1.0, 0.5;

    // The same rotation matrix as before; tetha radians arround Z axis
    transform_2.rotate (yawAngle*pitchAngle *rollAngle );
    std::cout<<transform_2.matrix() <<std::endl;
    std::cout<<transform_2.translation()<<std::endl;
    std::cout<<transform_2.translation().x()<<std::endl;
    std::cout<<transform_2.translation().y()<<std::endl;
    std::cout<<transform_2.translation().z()<<std::endl;

}

void determiningRollPitchYawFromRotationMatrix()
{
    /*  http://planning.cs.uiuc.edu/node103.html

      |r11 r12 r13 |
      |r21 r22 r23 |
      |r31 r32 r33 |

      yaw: alpha=arctan(r21/r11)
      pitch: beta=arctan(-r31/sqrt( r32^2+r33^2 ) )
      roll: gamma=arctan(r32/r33)
  */
    double roll, pitch, yaw;
    roll = M_PI / 2;
    pitch = M_PI / 2;
    yaw = 0;
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d rotationMatrix = q.matrix();

    std::cout << "Rotation Matrix is:" << std::endl;
    std::cout << rotationMatrix << std::endl;

    std::cout << "roll is Pi/"
              << M_PI / atan2(rotationMatrix(2, 1), rotationMatrix(2, 2))
              << std::endl;
    std::cout << "pitch: Pi/"
              << M_PI / atan2(-rotationMatrix(2, 0),
                            std::pow(
                                rotationMatrix(2, 1) * rotationMatrix(2, 1) + rotationMatrix(2, 2) * rotationMatrix(2, 2),
                                0.5))
              << std::endl;
    std::cout << "yaw is Pi/"
              << M_PI / atan2(rotationMatrix(1, 0), rotationMatrix(0, 0))
              << std::endl;
}

//unaryExpr, Lambda Expression, function pointer ,in place update
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

void maskingArray()
{
    //Eigen::MatrixXf P, Q, R; // 3x3 float matrix.

    // (R.array() < s).select(P,Q ); // (R < s ? P : Q)
    // R = (Q.array()==0).select(P,R); // R(Q==0) = P(Q==0)
    int cols, rows;
    cols=2; rows=3;
    Eigen::MatrixXf R=Eigen::MatrixXf::Random(rows, cols);

    Eigen::MatrixXf Q=Eigen::MatrixXf::Zero(rows, cols);
    Eigen::MatrixXf P=Eigen::MatrixXf::Constant(rows, cols,1.0);

    double s=0.5;
    Eigen::MatrixXf masked=(R.array() < s).select(P,Q ); // (R < s ? P : Q)

    std::cout<< R <<std::endl;
    std::cout<< masked <<std::endl;


    //ArrayXXd arrayA = ArrayXXd::Random(3, 2);
    //ArrayXXi mask = ArrayXXi::Zero(arrayA.rows(), arrayA.cols());
    //// mask = arrayA > 5;
    //// ArrayXd arrayB = arrayA(mask)

    ////(arrayA > 5).select(mask, arrayA);

    ///*
    //std::cout<< arrayA <<std::endl;
    //std::cout<<(arrayA > 0.05).select(mask, arrayA) <<std::endl;
    //std::cout<<mask <<std::endl;*/
}

void matrixReshaping()
{
    //https://eigen.tuxfamily.org/dox/group__TutorialReshapeSlicing.html
    /*
    Eigen::MatrixXd m1(12,1);
    m1<<0,1,2,3,4,5,6,7,8,9,10,11;
    std::cout<<m1<<std::endl;

    //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> m2(m1);
    //Eigen::Map<Eigen::MatrixXd> m3(m2.data(),3,4);
    Eigen::Map<Eigen::MatrixXd> m2(m1.data(),4,3);
    std::cout<<m2.transpose()<<std::endl;
    //solution*/
    //https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
}

void MatrixPowersPolynomials()
{

    //Rectangular Diagonal
    // rectangular diagonal  matrix is an n × d matrix in which each entry (i, j) has a non-zero value if and  only if i = j.
    // diagonal matrix is a matrix in which the entries outside the main diagonal are all zero;


    //A block diagonal matrix contains square blocks B 1…B r of (possibly) nonzero entries along the diagonal. All other entries are zero. Although each
    //block is square, they need not be of the same size.

    //Upper and Lower Triangular Matrix) A square matrix is an upper triangular matrix if all entries (i, j) below its main diagonal (i.e.,
    //satisfying i > j) are zeros.

    // The product of uppertriangular matrices is upper triangular.
    //c(i,j)=0 if i>j c(i,j)=sum(a(i,k)*b(k,j))
    //i>j
    //if i>k a(i,k)=0
    //if i<k -> j<k -> b(k,j)=0

    //Inverse of Triangular Matrix Is Triangula

    //Strictly Triangular Matrix) A matrix is said to be strictlytriangular if it is triangularandall its diagonal elements are zeros.

    //The zeroth power of a matrix is defined to be the identity matrix
    //When a matrix satisfies A^k = 0 for some integer k, it is referred to as nilpotent.
    //all strictly triangular matrices (triangular with zero main diagonal)of size d × d satisfy A^d = 0.
    //product of two upper triangular is a triangular matrix

    //polynomial function f(A) of a square matrix in  much the same way as one computes polynomials of scalars.
    //f(x) = 3x^2 + 5x + 2 -> f(A) = 3A^2 + 5A + 2
    //Two    polynomials f(A) and g(A) of the same matrix A will always commute f(A)g(A)=g(A)f(A)

    //Commutativity of Multiplication with Inverse if AB=I then BA=I
    //When the inverse of a matrix exists, it is always unique
    //Inverse of Triangular Matrix Is Triangular
    //Inv(A^n)=Inv(A)^n

    //An orthogonal matrix is a square matrix whose inverse is its transpose: A*A^T=A^T*A=I
    //all column columns/ rows are perpendicular

    //The multiplication of an n × d matrix A with a d-dimensional column
    //vector to create an n-dimensional column vector is often interpreted as
    //a linear transformation from d-dimensional space to n-dimensional space
/*
    a11   a12         a11      a12
    a21   a22  x1 =x1 a21 + x2 a22
    a31   a32  x2     a31      a32

Therefore, the n × d matrix A is occasionally represented in terms of its ordered set of ndimensional columns
A nxd= [a1 a2 ... ad]

 low-rank update
 Linear regression least-squares classification, support-vector machines, and logistic regression
 Matrix factorization is an alternative term for matrix decomposition
 recommender systems

Regression
the only difference from classification is that the array contains numerical values (rather than categorical ones)
The dependent
variable is also referred to as a response variable, target variable, or regressand in the case of regression
The independent variables are also referred to as regressors.more than two classes like {Red,
Green, Blue} cannot be ordered, and are therefore different from regression


convex objective functions like linear regression

https://medium.com/@wisnutandaseru/proving-eulers-identity-using-taylor-series-2771089cd780
Householder Reflections
Directional derivative
chain rule for multivariale derivative
multivariant fourier transform



inverse of I+A?
https://math.stackexchange.com/questions/298616/what-is-inverse-of-ia/298623
https://en.wikipedia.org/wiki/Woodbury_matrix_identity


https://en.m.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra
*/

}

void transformExample()
{
/*
The class Eigen::Transform  represents either
1) an affine, or
2) a projective transformation
using homogenous calculus.

For instance, an affine transformation A is composed of a linear part L and a translation t such that transforming a point p by A is equivalent to:

p' = L * p + t

Using homogeneous vectors:

    [p'] = [L t] * [p] = A * [p]
    [1 ]   [0 1]   [1]       [1]

Ref: https://stackoverflow.com/questions/35416880/what-does-transformlinear-return-in-the-eigen-library

Difference Between Projective and Affine Transformations
1) The projective transformation does not preserve parallelism, length, and angle. But it still preserves collinearity and incidence.
2) Since the affine transformation is a special case of the projective transformation,
it has the same properties. However unlike projective transformation, it preserves parallelism.

Ref: https://www.graphicsmill.com/docs/gm5/Transformations.htm
*/

    float arrVertices [] = { -1.0 , -1.0 , -1.0 ,
    1.0 , -1.0 , -1.0 ,
    1.0 , 1.0 , -1.0 ,
    -1.0 , 1.0 , -1.0 ,
    -1.0 , -1.0 , 1.0 ,
    1.0 , -1.0 , 1.0 ,
    1.0 , 1.0 , 1.0 ,
    -1.0 , 1.0 , 1.0};
    Eigen::MatrixXf mVertices = Eigen::Map < Eigen::Matrix <float , 3 , 8 > > ( arrVertices ) ;
    Eigen::Transform <float , 3 , Eigen::Affine > t = Eigen::Transform <float , 3 , Eigen::Affine >::Identity();
    t.scale ( 0.8f ) ;
    t.rotate ( Eigen::AngleAxisf (0.25f * M_PI , Eigen::Vector3f::UnitX () ) ) ;
    t.translate ( Eigen::Vector3f (1.5 , 10.2 , -5.1) ) ;
    std::cout << t * mVertices.colwise().homogeneous () << std::endl;
}

Eigen::Matrix4f createAffinematrix(float a, float b, float c, Eigen::Vector3f trans)
{
    {
        Eigen::Transform<float, 3, Eigen::Affine> t;
        t = Eigen::Translation<float, 3>(trans);
        t.rotate(Eigen::AngleAxis<float>(a, Eigen::Vector3f::UnitX()));
        t.rotate(Eigen::AngleAxis<float>(b, Eigen::Vector3f::UnitY()));
        t.rotate(Eigen::AngleAxis<float>(c, Eigen::Vector3f::UnitZ()));
        return t.matrix();
    }


    {
    /*
    The difference between the first implementation and the second is like the difference between "Fix Angle" and "Euler Angle", you can
    https://www.youtube.com/watch?v=09xVHo1JudY
    */
        Eigen::Transform<float, 3, Eigen::Affine> t;
        t = Eigen::AngleAxis<float>(c, Eigen::Vector3f::UnitZ());
        t.prerotate(Eigen::AngleAxis<float>(b, Eigen::Vector3f::UnitY()));
        t.prerotate(Eigen::AngleAxis<float>(a, Eigen::Vector3f::UnitX()));
        t.pretranslate(trans);
        return t.matrix();
    }
}

////////////////////////////C++ Functor////////////////////////////
/*** print the name of some types... ***/

template<typename type>
std::string name_of_type()
{
    return "other";
}

template<>
std::string name_of_type<int>()
{
    return "int";
}

template<>
std::string name_of_type<float>()
{
    return "float";
}

template<>
std::string name_of_type<double>()
{
    return "double";
}

template<typename scalar>
struct product_functor
{
    product_functor(scalar a, scalar b) : m_a(a), m_b(b)
    {
        std::cout << "Type: " << name_of_type<scalar>() << ". Computing the product of " << a << " and " << b << ".";
    }
    // the objective function a*b
    scalar f() const
    {
        return m_a * m_b;
    }

private:
    scalar m_a, m_b;
};

struct sum_of_ints_functor
{
    sum_of_ints_functor(int a, int b) : m_a(a), m_b(b)
    {
        std::cout << "Type: int. Computing the sum of the two ints " << a << " and " << b << ".";
    }

    int f() const
    {
        return m_a + m_b;
    }

    private:
    int m_a, m_b;
};

template<typename functor_type>
void call_and_print_return_value(const functor_type& functor_object)
{
    std::cout << " The result is: " << functor_object.f() << std::endl;
}

void functorExample()
{
    call_and_print_return_value(sum_of_ints_functor(3,5));
    call_and_print_return_value(product_functor<float>(0.2f,0.4f));
}


////////////////////////////Eigen Functor////////////////////////////


// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    // Information that tells the caller the numeric type (eg. double) and size (input / output dim)
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
};
// Tell the caller the matrix sizes associated with the input, output, and jacobian
typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

// Local copy of the number of inputs
int m_inputs, m_values;

// Two constructors:
Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

// Get methods for users to determine function input and output dimensions
int inputs() const { return m_inputs; }
int values() const { return m_values; }

};

///////////////////////////////////// Levenberg Marquardt Examples /////////////////////////////////////


// y = 10*(x0+3)^2 + (x1-5)^2
struct simpleFunctor : Functor<double>
{
    simpleFunctor(void): Functor<double>(2,2)
    {

    }
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        // Implement y = 10*(x0+3)^2 + (x1-5)^2
        //fvec(0) = 10.0*pow(x(0)+3.0,2) +  pow(x(1)-5.0,2);
        //fvec(1) = 0;
        fvec(0) = 10.0*pow(x(0)+3.0,2) ;
        fvec(1) = pow(x(1)-5.0,2);
        return 0;
    }
};

void simpleFunctorExample()
{
    Eigen::VectorXd x(2);
    x(0) = 2.0;
    x(1) = 3.0;
    std::cout << "starting x: \n" << x << std::endl;

    simpleFunctor functor;
    Eigen::NumericalDiff<simpleFunctor> numDiff(functor);

    Eigen::MatrixXd fjac(2,2);
    numDiff.df(x,fjac);


    std::cout << "jacobian of matrix at "<< x(0)<<","<<x(1)  <<" is:\n " << fjac << std::endl;

    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<simpleFunctor>,double> lm(numDiff);
    lm.parameters.maxfev = 2000;
    lm.parameters.xtol = 1.0e-10;
    std::cout <<"maximum number of function evaluation: " <<lm.parameters.maxfev << std::endl;

    int ret = lm.minimize(x);
    std::cout <<"number of iterations: " <<lm.iter << std::endl;

    std::cout << "x that minimizes the function: \n" << x << std::endl;

    //std::cout << "optimized value: "<<ret << std::endl;

    std::cout << "press [ENTER] to continue " << std::endl;
    std::cin.get();

}


// https://en.wikipedia.org/wiki/Test_functions_for_optimization
// Booth Function
// Implement f(x,y) = (x + 2*y -7)^2 + (2*x + y - 5)^2
struct BoothFunctor : Functor<double>
{
    // Simple constructor
    BoothFunctor(): Functor<double>(2,2) {}

    // Implementation of the objective function
    int operator()(const Eigen::VectorXd &z, Eigen::VectorXd &fvec) const
    {
        double x = z(0);   double y = z(1);
        /*
        * Evaluate the Booth function.
        * Important: LevenbergMarquardt is designed to work with objective functions that are a sum
        * of squared terms. The algorithm takes this into account: do not do it yourself.
        * In other words: objFun = sum(fvec(i)^2)
        */
        fvec(0) = x + 2*y - 7;
        fvec(1) = 2*x + y - 5;
        return 0;
    }
};


void testBoothFun() {
    std::cout << "Testing the Booth function..." << std::endl;
    Eigen::VectorXd zInit(2); zInit << 1.87, 2.032;
    std::cout << "zInit: " << zInit.transpose() << std::endl;
    Eigen::VectorXd zSoln(2); zSoln << 1.0, 3.0;
    std::cout << "zSoln: " << zSoln.transpose() << std::endl;

    BoothFunctor functor;
    Eigen::NumericalDiff<BoothFunctor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<BoothFunctor>,double> lm(numDiff);
    lm.parameters.maxfev = 1000;
    lm.parameters.xtol = 1.0e-10;
    std::cout << "max fun eval: " << lm.parameters.maxfev << std::endl;
    std::cout << "x tol: " << lm.parameters.xtol << std::endl;

    Eigen::VectorXd z = zInit;
    int ret = lm.minimize(z);
    std::cout << "iter count: " << lm.iter << std::endl;
    std::cout << "return status: " << ret << std::endl;
    std::cout << "zSolver: " << z.transpose() << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
}


struct simpleMultiPolynomialFunctor : Functor<double>
{
    // Simple constructor
    simpleMultiPolynomialFunctor(): Functor<double>(3,2) {}

    // Implementation of the objective function
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {

        fvec(0) = 2*pow(x(0),2) + 5*x(1) +pow(x(2),3);
        fvec(1) = 3*x(0) + 2*pow(x(1),3)+ x(1)*x(2);

        return 0;
    }
};


void numericalDifferentiationExample()
{
    int diffMode= Eigen::NumericalDiffMode::Central; //Eigen::NumericalDiffMode::Central


    simpleMultiPolynomialFunctor functor;
    Eigen::NumericalDiff<simpleMultiPolynomialFunctor,Eigen::NumericalDiffMode::Central> numDiff(functor);


    Eigen::VectorXd x(3);
    x(0) = -1.0;
    x(1) = 1.0;
    x(2) = 1.0;

    Eigen::MatrixXd fjac(2,3);
    numDiff.df(x,fjac);

    std::cout << "numerical differentiation at \n"<<x <<"\nis: \n" << fjac << std::endl;
}

/*

Refs
1) https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/LevenbergMarquardt/CurveFitting.cpp
2) https://stackoverflow.com/questions/18509228/how-to-use-the-eigen-unsupported-levenberg-marquardt-implementation
https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/LevenbergMarquardt/CurveFitting.cpp
https://stackoverflow.com/questions/48213584/understanding-levenberg-marquardt-enumeration-returns
https://www.ultimatepp.org/reference$Eigen_demo$en-us.html
https://ethz-adrl.github.io/ct/ct_doc/doc/html/core_tut_linearization.html

*/


///////////////////////////////////////////////// AutoDiffScalar /////////////////////////////////////////////////
/*
Refs:
https://joelcfd.com/automatic-differentiation/
*/

int main(int argc, char *argv[])
{
    //simpleFunctorExample();
    numericalDifferentiationExample();
    return 0;
}

