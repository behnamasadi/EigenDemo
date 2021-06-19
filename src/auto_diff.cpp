/*
Refs: https://joelcfd.com/automatic-differentiation/
*/

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

template<typename T>
T MultiVarFunction(const T &x,const  T& y)
{
    T z=x*cos(y);
    return z;
}

void MultiVarFunctionDerivativesExample()
{
    Eigen::AutoDiffScalar<Eigen::VectorXd> x,y,z_derivative;

    x.value()=2  ;
    y.value()=M_PI/4;

/*
    Eigen::VectorXd::Unit(2, 0) means a unit vector with first element be 1

    0
    1

    which means we want compute the derivates relative to the first element
*/
    //we telling we want dz/dx
    x.derivatives() = Eigen::VectorXd::Unit(2, 0);

    //we telling we want dz/dy
    y.derivatives() = Eigen::VectorXd::Unit(2, 1);

    z_derivative = MultiVarFunction(x, y);

    std::cout << "AutoDiff:" << std::endl;
    std::cout << "Function output: " << z_derivative.value() << std::endl;
    std::cout << "Derivative: \n" << z_derivative.derivatives()<< std::endl;

}

template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct VectorFunction {

    typedef _Scalar Scalar;

    enum
    {
      InputsAtCompileTime = NX,
      ValuesAtCompileTime = NY
    };

    // Also needed by Eigen
    typedef Eigen::Matrix<Scalar, NX, 1> InputType;
    typedef Eigen::Matrix<Scalar, NY, 1> ValueType;

    // Vector function
    template <typename Scalar>
    void operator()(const Eigen::Matrix<Scalar, NX, 1>& x, Eigen::Matrix<Scalar, NY, 1>* y ) const
    {
        (*y)(0,0) = 10.0*pow(x(0,0)+3.0,2) +pow(x(1,0)-5.0,2) ;
        (*y)(1,0) = (x(0,0)+1)*x(1,0);
        (*y)(2,0) = sin(x(0,0))*x(1,0);
    }
};



void JacobianDerivativesExample()
{

    Eigen::Matrix<double, 2, 1> x;
    Eigen::Matrix<double, 3, 1> y;
    Eigen::Matrix<double, 3,2> fjac;

    Eigen::AutoDiffJacobian< VectorFunction<double,2, 3> > JacobianDerivatives;

    // Set values in x, y and fjac...


    x(0,0)=2.0;
    x(1,0)=3.0;

    JacobianDerivatives(x, &y, &fjac);

    std::cout << "jacobian of matrix at "<< x(0,0)<<","<<x(1,0)  <<" is:\n " << fjac << std::endl;

}

int main()
{
    MultiVarFunctionDerivativesExample();
    JacobianDerivativesExample();
}
