#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <kdl/frames.hpp>
using namespace Eigen;
using namespace std;

//	c complex
//	d double
//	i int
//	f float
//  for more : http://eigen.tuxfamily.org/dox/group__matrixtypedefs.html

void VectorsExample()
{
//	Vectors
	//Vector3f is a (column) vector of 3 floats:
	Vector3f objVector3f;

	//row-vectors:
	RowVector2i objRowVector2i;

	VectorXf v(10);


}

void FixedSizedMatrixMatrix()
{

//	c complex
//	d double
//	i int
//	f float
//  for more : http://eigen.tuxfamily.org/dox/group__matrixtypedefs.html
	Matrix4d m; //4x4 double
	cout<<m.rows() <<endl;
	cout<<m.cols()<<endl;

	Matrix4cd objMatrix4cd; //4x4 double complex
	cout<<objMatrix4cd.rows() <<endl;
	cout<<objMatrix4cd.cols()<<endl;
}

void DynamicSizeMatrix()
{

//	a is a 3x3 matrix, with a static float[9] array of uninitialized coefficients,
//	b is a dynamic-size matrix whose size is currently 0x0, and whose array of coefficients hasn't yet been allocated at all.

	Matrix3f a;
	MatrixXf b;


//	a is a 10x15 dynamic-size matrix, with allocated but currently uninitialized coefficients.
//	b is a dynamic-size vector of size 30, with allocated but currently uninitialized coefficients.
	MatrixXf A(10,15);
	VectorXf v(30);


	A.resize(4,3);
	std::cout << "The matrix A is of size " << A.rows() << "x" << A.cols() << std::endl;
	std::cout << "It has " << A.size() << " coefficients" << std::endl;


	v.resize(5);
	std::cout << "The vector v is of size " << v.size() << std::endl;
	std::cout << "As a matrix, v is of size " << v.rows() << "x" << v.cols() << std::endl;

	for(int i=0;i<A.rows();i++)
	{
		for(int j=0;j<A.cols();j++)
		{
			A(i,j)=i+j;
		}
	}
    cout<<"A is : " <<A <<endl;

    Eigen::Matrix2d a_2d;
    a_2d.setRandom();
    std::cout<<"a_2d: " <<a_2d <<std::endl;
//    setOnes()
//    setConstant()
//    setRandom()
//    setLinSpaced()


    std::cout<<"a_2d.minCoeff(): " <<a_2d.minCoeff() <<std::endl;
    std::cout<<"a_2d.maxCoeff(): " <<a_2d.maxCoeff() <<std::endl;

//    m.minCoeff()
//    m.maxCoeff()
//    m.prod()
//    m.sum()
//    m.mean()
//    m.trace()
//    m.colwise()
//    m.rowwise()
//    m.all()
//    m.any()
//    m.lpNorm<p>()
//    m.lpNorm<Infinity>()

}

void EigenValueSolver()
{
	  //ComplexEigenSolver<MatrixXcf> ces;
	  EigenSolver<MatrixXd> ces;
	  MatrixXd A(4,4);
	  A(0,0)= 18;
	  A(0,1)=-9;
	  A(0,2)=-27;
	  A(0,2)=24;
	  A(1,0)=-9;
	  A(1,1)= 4.5;
	  A(1,2)=13.5;
	  A(1,3)=-12 ;
	  A(2,0)=-27;
	  A(2,1)=13.5;
	  A(2,2)=40.5;
	  A(2,3)=-36;
	  A(3,0)=24;
	  A(3,1)= -12;
	  A(3,2)=-36;
	  A(3,3)=32;
	  ces.compute(A);
	  cout << "The eigenvalues of A are:" << endl << ces.eigenvalues() << endl;
	  cout << "The matrix of eigenvectors, V, is:" << endl << ces.eigenvectors() << endl << endl;
	//  complex<float> lambda =  ces.eigenvalues()[0];
	//  cout << "Consider the first eigenvalue, lambda = " << lambda << endl;
	//  VectorXcf v = ces.eigenvectors().col(0);
	//  cout << "If v is the corresponding eigenvector, then lambda * v = " << endl << lambda * v << endl;
	//  cout << "... and A * v = " << endl << A * v << endl << endl;
	//
	//  cout << "Finally, V * D * V^(-1) = " << endl
	//       << ces.eigenvectors() * ces.eigenvalues().asDiagonal() * ces.eigenvectors().inverse() << endl;
}


void Transformation()
{
    //If you are working with OpenGL 4x4 matrices then Affine3f and Affine3d are what you want.
    //Since Eigen defaults to column-major storage, you can directly use the Transform::data() method to pass your transformation matrix to OpenGL.

    //construct a Transform:
    //	Transform t(AngleAxis(angle,axis));
    //or like this:
    //	Transform t;
    //	t = AngleAxis(angle,axis);


    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //But note that unfortunately, because of how C++ works, you can not do this:
    //  Transform t = AngleAxis(angle,axis);
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//

/*

yaw:
    A yaw is a counterclockwise rotation of alpha about the  z-axis. The rotation matrix is given by

    R_z

    |cos(alpha) -sin(alpha) 0|
    |sin(apha)   cos(alpha) 0|
    |    0            0     1|

pitch:
    R_y
    A pitch is a counterclockwise rotation of  beta about the  y-axis. The rotation matrix is given by

    |cos(beta)  0   sin(beta)|
    |0          1       0    |
    |-sin(beta) 0   cos(beta)|

roll:
    A roll is a counterclockwise rotation of  gamma about the  x-axis. The rotation matrix is given by
    R_x
    |1          0           0|
    |0 cos(gamma) -sin(gamma)|
    |0 sin(gamma)  cos(gamma)|



    It is important to note that   R_z R_y R_x performs the roll first, then the pitch, and finally the yaw
    Roration matrix: R_z*R_y*R_x





*/




    double roll, pitch, yaw;
    roll=M_PI/3;
    pitch=M_PI/4;
    yaw=M_PI/6;
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());




    Eigen::Quaternion<double> q =  yawAngle*pitchAngle *rollAngle;

    Eigen::Matrix3d rotationMatrix = q.matrix();
    std::cout<<rotationMatrix <<std::endl;



    KDL::Frame F;
    F.M=F.M.RPY(roll, pitch, yaw);
    std::cout<<F.M(0,0) <<" " <<F.M(0,1)<<" " <<F.M(0,2) <<std::endl;
    std::cout<<F.M(1,0) <<" " <<F.M(1,1)<<" " <<F.M(1,2) <<std::endl;
    std::cout<<F.M(2,0) <<" " <<F.M(2,1)<<" " <<F.M(2,2) <<std::endl;





    Eigen::Affine3d transform_2 = Eigen::Affine3d::Identity();

    // Define a translation of 2.5 meters on the x axis.
    transform_2.translation() << 2.5, 0.0, 0.0;

    // The same rotation matrix as before; tetha radians arround Z axis
    transform_2.rotate (yawAngle*pitchAngle *rollAngle );
    std::cout<<transform_2.matrix() <<std::endl;
}

void DeterminingRollPitchYaw()
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
    roll=M_PI/3;
    pitch=M_PI/4;
    yaw=M_PI/6;
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaternion<double> q =  yawAngle*pitchAngle *rollAngle;
    Eigen::Matrix3d rotationMatrix = q.matrix();



    std::cout<<"yaw is Pi/" <<M_PI/atan2( rotationMatrix(1,0),rotationMatrix(0,0) ) <<std::endl;
//    std::cout<<"pitch: " << <<std::endl;
    std::cout<<"roll is Pi/" <<M_PI/atan2( rotationMatrix(2,1),rotationMatrix(2,2) ) <<std::endl;



}

void define_arbitrary_matrix()
{
//    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    Eigen::Matrix< double ,2,3> my_matrix;
    my_matrix<<1,2,3,4,5,6;
    std::cout<< my_matrix<<std::endl;
}

void ArrayExample()
{
/*The Array class provides general-purpose arrays, as opposed to the Matrix class which is intended for linear algebra.
 * Furthermore, the Array class provides an easy way to perform coefficient-wise operations, which might not have a linear algebraic meaning, such as adding a constant to every coefficient in the array or multiplying two arrays coefficient-wise.*/
    //ArrayXf
    Array<float,Dynamic,1> a1;
    //Array3f
    Array<float,3,1> a2;
    //ArrayXXd
    Array<double,Dynamic,Dynamic> a3;
    //Array33d
    Array<double,3,3> a4;
    Eigen::Matrix3d matrix_from_array = a4.matrix();


}

void CoefficientWiseOperation()
{
    /*Must convert Matrix<T> to Array<T> using array() member     function in order to perform coefficient-wise operations*/
    Eigen::Matrix2i m;
    m<<-1,2,3,4;

    Eigen::Matrix2i a;
    a<<2,2,3,4;
    std::cout<<"m: " << m<<std::endl;
    std::cout<<"m.array() -2 " << m.array() -2<<std::endl;
    std::cout<<"m.array().abs() " <<m.array().abs()<<std::endl;
    std::cout<<"m.array().square() " <<m.array().square()<<std::endl;
    std::cout<<"m.array()*a.array() " <<m.array()*a.array()<<std::endl;
//    pow()
//    exp()
//    log()
//    sqrt()
}

void EigenMapExample()
{
    int array[12];
    for(int i = 0; i < 12; ++i)
    {
         array[i] = i;
    }
    Eigen::MatrixXi a;
    a= Map<MatrixXi,0,OuterStride<> >(array,6, 2, OuterStride<>(6));
    std::cout<<a <<std::endl;


    int * a_Matrix_ptr=a.data();

    for(std::size_t i =0;i<a.rows()*a.cols();i++)
    {
        std::cout<<a_Matrix_ptr[i]<<std::endl;
    }


}

void LinearSolving()
{
    //Ax=b, x=?
    MatrixXf A(3,3);
    VectorXf b(3,1);
    A << 1,2,3,  4,5,6,  7,8,10;
    b << 3, 3, 4;
    cout << "Here is the matrix A:\n" << A << endl;
    cout << "Here is the vector b:\n" << b << endl;
    VectorXf x = A.colPivHouseholderQr().solve(b);
    cout << "The solution using colPivHouseholderQr is:\n" << x << endl;



    cout << "The solution using partialPivLu() is:\n" << A.partialPivLu().solve(b) << endl;

/*
    partialPivLu()
    fullPivLu()
    householderQr()
    colPivHouseholderQr()
    fullPivHouseholderQr()
    llt()
    ldlt()
*/

}

void InverseOfMatrix()
{
    Matrix3f A;
    A << 1, 2, 1,
         2, 1, 0,
         -1, 1, 2;
    cout << "Here is the matrix A:\n" << A << endl;
    cout << "The determinant of A is " << A.determinant() << endl;
    cout << "The inverse of A is:\n" << A.inverse() << endl;

}

int main()
{
//    FixedSizedMatrixMatrix();
//    DynamicSizeMatrix();
//    EigenMapExample();
//    CoefficientWiseOperation();
//    define_arbitrary_matrix();
//    LinearSolving();
//    InverseOfMatrix();
//    Transformation();
    DeterminingRollPitchYaw();
}
