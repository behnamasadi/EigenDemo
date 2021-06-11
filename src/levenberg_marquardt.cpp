#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
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

///////////////////////// Levenberg Marquardt Example (1) f(x) = a x + b ///////////////////////////////

typedef std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > Point2DVector;

Point2DVector GeneratePoints();

struct SimpleLineFunctor : Functor<double>
{
    int operator()(const Eigen::VectorXd &z, Eigen::VectorXd &fvec) const
    {
        // "a" in the model is z(0), and "b" is z(1)
        double x_i,y_i,a,b;
        for(unsigned int i = 0; i < this->Points.size(); ++i)
        {
            y_i=this->Points[i](1);
            x_i=this->Points[i](0);
            a=z(0);
            b=z(1);
            fvec(i) =y_i-(a*x_i +b);
        }

        return 0;
    }

  Point2DVector Points;

  int inputs() const { return 2; } // There are two parameters of the model
  int values() const { return this->Points.size(); } // The number of observations
};

struct SimpleLineFunctorNumericalDiff : Eigen::NumericalDiff<SimpleLineFunctor> {};

Point2DVector GeneratePoints(const unsigned int numberOfPoints)
{
    Point2DVector points;
    // Model y = 2*x + 5 with some noise (meaning that the resulting minimization should be about (2,5)
    for(unsigned int i = 0; i < numberOfPoints; ++i)
    {
        double x = static_cast<double>(i);
        Eigen::Vector2d point;
        point(0) = x;
        point(1) = 2.0 * x + 5.0 + drand48()/10.0;
        points.push_back(point);
    }

  return points;
}

void testSimpleLineFunctor()
{
    std::cout << "Testing f(x) = a x + b function..." << std::endl;

    unsigned int numberOfPoints = 50;
    Point2DVector points = GeneratePoints(numberOfPoints);

    Eigen::VectorXd x(2);
    x.fill(2.0f);

    SimpleLineFunctorNumericalDiff functor;
    functor.Points = points;
    Eigen::LevenbergMarquardt<SimpleLineFunctorNumericalDiff> lm(functor);

    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
    std::cout << "status: " << status << std::endl;
    std::cout << "x that minimizes the function: " << std::endl << x << std::endl;
    std::cout <<"lm.parameters.epsfcn: " <<lm.parameters.epsfcn << std::endl;
    std::cout <<"lm.parameters.factor: " <<lm.parameters.factor << std::endl;
    std::cout <<"lm.parameters.ftol: " << lm.parameters.ftol << std::endl;
    std::cout <<"lm.parameters.gtol: " <<lm.parameters.gtol << std::endl;
    std::cout <<"lm.parameters.maxfev: " <<lm.parameters.maxfev << std::endl;
    std::cout <<"lm.parameters.xtol: " <<lm.parameters.xtol << std::endl;

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

}

///////////////////////// Levenberg Marquardt Example (2) f(x) = ax² + bx + c ///////////////////////////////


// Ref: https://medium.com/@sarvagya.vaish/levenberg-marquardt-optimization-part-2-5a71f7db27a0

struct QuadraticFunctor
{
    // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        // 'x' has dimensions n x 1
        // It contains the current estimates for the parameters.

        // 'fvec' has dimensions m x 1
        // It will contain the error for each data point.

        double aParam = x(0);
        double bParam = x(1);
        double cParam = x(2);

        for (int i = 0; i < values(); i++)
        {
            double xValue = measuredValues(i, 0);
            double yValue = measuredValues(i, 1);

            fvec(i) = yValue - (aParam * xValue * xValue + bParam * xValue + cParam);
        }
    }

    // Compute the jacobian of the errors
    int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
    {
        // 'x' has dimensions n x 1
        // It contains the current estimates for the parameters.

        // 'fjac' has dimensions m x n
        // It will contain the jacobian of the errors, calculated numerically in this case.

        float epsilon;
        epsilon = 1e-5f;

        for (int i = 0; i < x.size(); i++)
        {
            Eigen::VectorXd xPlus(x);
            xPlus(i) += epsilon;
            Eigen::VectorXd xMinus(x);
            xMinus(i) -= epsilon;

            Eigen::VectorXd fvecPlus(values());
            operator()(xPlus, fvecPlus);

            Eigen::VectorXd fvecMinus(values());
            operator()(xMinus, fvecMinus);

            Eigen::VectorXd fvecDiff(values());
            fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

            fjac.block(0, i, values(), 1) = fvecDiff;
        }
    }

    // Number of data points, i.e. values.
    int m;

    // Returns 'm', the number of values.
    int values() const { return m; }

    // The number of parameters, i.e. inputs.
    int n;

    // Returns 'n', the number of inputs.
    int inputs() const { return n; }

    Eigen::MatrixXd measuredValues;
};

void quadraticPointsGenerator(std::vector<double> &x_values, std::vector<double> &y_values,double a=-1, double b=0.6, double c=2, unsigned int numberOfPoints=50 )
{
    //point are x_start=-4, x_end=3
    double x_start=-4;
    double x_end=3;
    double x,y;

    for(unsigned int i = 0; i < numberOfPoints; ++i)
    {
        x = x_start+ static_cast<double>(i)* ( x_end- x_start)/numberOfPoints;
        y = a*pow(x,2)+b*x+c + drand48()/10.0;
        x_values.push_back(x);
        y_values.push_back(y);
    }

}

void testQuadraticFunctor()
{
    std::cout << "Testing the f(x) = ax² + bx + c function..." << std::endl;
    std::vector<double> x_values;
    std::vector<double> y_values;

    double a=-1;
    double b=0.6;
    double c=2;

    quadraticPointsGenerator(x_values, y_values,a,b,c);

    // 'm' is the number of data points.
    int m = x_values.size();

    // Move the data into an Eigen Matrix.
    // The first column has the input values, x. The second column is the f(x) values.
    Eigen::MatrixXd measuredValues(m, 2);
    for (int i = 0; i < m; i++) {
        measuredValues(i, 0) = x_values[i];
        measuredValues(i, 1) = y_values[i];
    }

    // 'n' is the number of parameters in the function.
    // f(x) = a(x^2) + b(x) + c has 3 parameters: a, b, c
    int n = 3;

    // 'parameters' is vector of length 'n' containing the initial values for the parameters.
    // The parameters 'x' are also referred to as the 'inputs' in the context of LM optimization.
    // The LM optimization inputs should not be confused with the x input values.
    Eigen::VectorXd parameters(n);
    parameters(0) = 0.0;             // initial value for 'a'
    parameters(1) = 0.0;             // initial value for 'b'
    parameters(2) = 0.0;             // initial value for 'c'

    //
    // Run the LM optimization
    // Create a LevenbergMarquardt object and pass it the functor.
    //

    QuadraticFunctor functor;
    functor.measuredValues = measuredValues;
    functor.m = m;
    functor.n = n;

    Eigen::LevenbergMarquardt<QuadraticFunctor, double> lm(functor);
    int status = lm.minimize(parameters);
    std::cout << "LM optimization status: " << status << std::endl;

    //
    // Results
    // The 'x' vector also contains the results of the optimization.
    //
    std::cout << "Optimization results" << std::endl;
    std::cout << "\ta: " << parameters(0) << std::endl;
    std::cout << "\tb: " << parameters(1) << std::endl;
    std::cout << "\tc: " << parameters(2) << std::endl;

    std::cout << "Actual values are" << std::endl;
    std::cout << "\ta: " << a << std::endl;
    std::cout << "\tb: " << b << std::endl;
    std::cout << "\tc: " << c << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

}


///////////////////////// Levenberg Marquardt Example (3) f(x,y) = (x + 2*y -7)^2 + (2*x + y - 5)^2 ///////////////////////////////


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

void testBoothFunctor() {
    std::cout << "Testing the f(x,y) = (x + 2*y -7)^2 + (2*x + y - 5)^2 function..." << std::endl;
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



///////////////////////// Levenberg Marquardt Example (4) f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 ///////////////////////////////


// https://en.wikipedia.org/wiki/Test_functions_for_optimization
// Himmelblau's Function
// Implement f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
struct HimmelblauFunctor : Functor<double>
{
    // Simple constructor
    HimmelblauFunctor(): Functor<double>(2,2) {}

    // Implementation of the objective function
    int operator()(const Eigen::VectorXd &z, Eigen::VectorXd &fvec) const
    {
        double x = z(0);   double y = z(1);
        /*
        * Evaluate Himmelblau's function.
        * Important: LevenbergMarquardt is designed to work with objective functions that are a sum
        * of squared terms. The algorithm takes this into account: do not do it yourself.
        * In other words: objFun = sum(fvec(i)^2)
        */
        fvec(0) = x * x + y - 11;
        fvec(1) = x + y * y - 7;
        return 0;
    }
};

void testHimmelblauFunctor()
{
    std::cout << "Testing f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 function..." << std::endl;
    // Eigen::VectorXd zInit(2); zInit << 0.0, 0.0;  // soln 1
    // Eigen::VectorXd zInit(2); zInit << -1, 1;  // soln 2
    // Eigen::VectorXd zInit(2); zInit << -1, -1;  // soln 3
    Eigen::VectorXd zInit(2); zInit << 1, -1;  // soln 4
    std::cout << "zInit: " << zInit.transpose() << std::endl;
    std::cout << "soln 1: [3.0, 2.0]" << std::endl;
    std::cout << "soln 2: [-2.805118, 3.131312]" << std::endl;
    std::cout << "soln 3: [-3.77931, -3.28316]" << std::endl;
    std::cout << "soln 4: [3.584428, -1.848126]" << std::endl;

    HimmelblauFunctor functor;
    Eigen::NumericalDiff<HimmelblauFunctor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<HimmelblauFunctor>,double> lm(numDiff);
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


int main()
{
    testSimpleLineFunctor();
    testQuadraticFunctor();
    testBoothFunctor();
    testHimmelblauFunctor();
}


/*

Refs
https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/LevenbergMarquardt/CurveFitting.cpp
https://stackoverflow.com/questions/18509228/how-to-use-the-eigen-unsupported-levenberg-marquardt-implementation
https://stackoverflow.com/questions/48213584/understanding-levenberg-marquardt-enumeration-returns
https://www.ultimatepp.org/reference$Eigen_demo$en-us.html
https://ethz-adrl.github.io/ct/ct_doc/doc/html/core_tut_linearization.html
https://robotics.stackexchange.com/questions/20673/why-with-the-pseudo-inverse-it-is-possible-to-invert-the-jacobian-matrix-even-in
http://users.ics.forth.gr/~lourakis/
https://mathoverflow.net/questions/257699/gauss-newton-vs-gradient-descent-vs-levenberg-marquadt-for-least-squared-method
https://math.stackexchange.com/questions/1085436/gauss-newton-versus-gradient-descent
https://stackoverflow.com/questions/34701160/how-to-set-levenberg-marquardt-damping-using-eigen
http://www.netlib.org/minpack/lmder.f
https://en.wikipedia.org/wiki/Test_functions_for_optimization
https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/LevenbergMarquardt/NumericalDerivative.cpp
https://stackoverflow.com/questions/18509228/how-to-use-the-eigen-unsupported-levenberg-marquardt-implementation
*/

