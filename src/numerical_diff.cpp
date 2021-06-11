#include <iostream>
#include <unsupported/Eigen/NumericalDiff>

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


// y0 = 10*(x0+3)^2 + (x1-5)^2
// y1 = (x0+1)*x1
// y2 = sin(x0)*x1
struct simpleFunctor : Functor<double>
{
    simpleFunctor(void): Functor<double>(2,3)
    {

    }
    // Implementation of the objective function

    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        fvec(0) = 10.0*pow(x(0)+3.0,2) +pow(x(1)-5.0,2) ;
        fvec(1) = (x(0)+1)*x(1);
        fvec(2) = sin(x(0))*x(1);
        return 0;
    }
};


void simpleNumericalDiffExample(Eigen::VectorXd &x)
{
    simpleFunctor functor;
    Eigen::NumericalDiff<simpleFunctor,Eigen::NumericalDiffMode::Central> numDiff(functor);
    Eigen::MatrixXd fjac(3,2);
    numDiff.df(x,fjac);
    std::cout << "jacobian of matrix at "<< x(0)<<","<<x(1)  <<" is:\n " << fjac << std::endl;
}


int main()
{
    Eigen::VectorXd x(2);
    x(0) = 2.0;
    x(1) = 3.0;
    simpleNumericalDiffExample(x);
}
