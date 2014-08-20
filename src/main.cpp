#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
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
	cout<<"A is : " <<A<<endl;
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

//	Affine Transformation
//	An affine transformation is equivalent to a linear transformation followed by a translation.
//	linear transformation:
//		rotation
//		reflection
//		scaling
//		horizontal shear mapping
//		squeeze mapping
//		projection


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

void AffineTransformation()
{
//	2D rotation from an angle
	float angle_in_radian=M_PI;
	Rotation2D<float> rot2( angle_in_radian);

//	3D rotation as an angle + axis(The axis vector must be normalized!)
	float ax,ay,az;
	AngleAxis<float> aa(angle_in_radian, Vector3f(ax,ay,az));


//	N-D Scaling
	float sx,sy,sz,s;
	int N=10;
	VectorXf vecN(N);
	Scaling(sx, sy);
	Scaling(sx, sy, sz);
	Scaling(s);
	Scaling(vecN);

//	N-D Translation
	float tx, ty,tz;
	Translation<float,2>(tx, ty);
	Translation<float,3>(tx, ty, tz);
//	Translation<float,N>(s);
//	Translation<float,N>(vecN);

//	N-D Affine transformation
//	Transform<float,N,Affine> t = concatenation_of_any_transformations;
//	Translation t;
//	t= Translation<float,3>(tx, ty, tz);
	Translation<float,3> t(tx, ty, tz);
	Transform<float,3,Affine> AffineTranslation = t * aa * Scaling(s);




}

void  Test()
{
//	When setting up an AngleAxis object, the axis vector must be normalized.
	Matrix3f m;
	m = AngleAxisf(0.25*M_PI, Vector3f::UnitX())  * AngleAxisf(0.5*M_PI,  Vector3f::UnitY())  * AngleAxisf(0.33*M_PI, Vector3f::UnitZ());
//	cout << m << endl << "is unitary: " << m.isUnitary() << endl;
//	cout<<"Vector3f::UnitX() " <<Vector3f::UnitX() <<endl;
//	cout<<"Vector3f::UnitY() " <<Vector3f::UnitY() <<endl;
//	cout<<"Vector3f::UnitZ() " <<Vector3f::UnitZ() <<endl;

////	If you are working with OpenGL 4x4 matrices then Affine3f and Affine3d are what you want.
////	typedef Transform< float,3, Affine > 	Affine3f
//	Affine3f m1;
//	cout<< m1.Rows<<endl;
//	cout<< m1.Dim <<endl;
//	cout<< m1.Mode <<endl;
////	m1 = AngleAxis3f(..);
////
////	Affine3f m2;
////	m2 = Scaling(..);

	Vector3f offset;
	Matrix4f  	transform;
	transform(0,0)=m(0,0);
	transform(0,1)=m(0,1);
	transform(0,2)=m(0,2);
	transform(1,0)=m(1,0);
	transform(1,1)=m(1,1);
	transform(1,2)=m(1,2);
	transform(2,0)=m(2,0);
	transform(2,1)=m(2,1);
	transform(2,2)=m(2,2);

	transform(0,3)=offset(0);
	transform(1,3)=offset(1);
	transform(2,3)=offset(2);

	transform(3,0)=0.0;
	transform(3,1)=0.0;
	transform(3,2)=0.0;
	transform(3,3)=1.0;


}


int main()
{
//	FixedSizedMatrixMatrix();
//	DynamicSizeMatrix();
	Test();


}
