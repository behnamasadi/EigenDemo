- [Jacobian](#jacobian)
- [Hessian Matrix](#hessian-matrix)
- [Automatic Differentiation](#automatic-differentiation)
- [Numerical Differentiation](#numerical-differentiation)

# Jacobian

Suppose <img  src="https://latex.codecogs.com/svg.latex?f%20:%20R_{n}%20\rightarrow%20R_{m}"  alt="https://latex.codecogs.com/svg.latex?f : R_{n} \rightarrow R_{m}" />. This function takes a point <img  src="https://latex.codecogs.com/svg.latex?x%20\in%20R_{n}" alt="https://latex.codecogs.com/svg.latex?x \in R_{n}" />  as input and produces the vector 
<img  src="https://latex.codecogs.com/svg.latex?f(x)%20\in%20R_{m}"  alt="https://latex.codecogs.com/svg.latex?f(x) \in R_{m}" /> as output. Then the Jacobian matrix of 
<img  src="https://latex.codecogs.com/svg.latex?f"  alt="https://latex.codecogs.com/svg.latex?f" />  is defined to be an m×n matrix, denoted by  <img  src="https://latex.codecogs.com/svg.latex?J"  alt="https://latex.codecogs.com/svg.latex?J" />  , whose <img  src="https://latex.codecogs.com/svg.latex?(i,j)_{th}"  alt="https://latex.codecogs.com/svg.latex?(i,j)_{th}" /> entry is: <img  src="https://latex.codecogs.com/svg.latex?{\textstyle%20\mathbf%20{J}%20_{ij}={\frac%20{\partial%20f_{i}}{\partial%20x_{j}}}}"  alt="https://latex.codecogs.com/svg.latex?{\textstyle \mathbf {J} _{ij}={\frac {\partial f_{i}}{\partial x_{j}}}}" /> or explicitly:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\mathbf%20{J}%20={\begin{bmatrix}{\dfrac%20{\partial%20\mathbf%20{f}%20}{\partial%20x_{1}}}&\cdots%20&{\dfrac%20{\partial%20\mathbf%20{f}%20}{\partial%20x_{n}}}\end{bmatrix}}={\begin{bmatrix}\nabla%20^{\mathrm%20{T}%20}f_{1}\\\vdots%20\\\nabla%20^{\mathrm%20{T}%20}f_{m}\end{bmatrix}}={\begin{bmatrix}{\dfrac%20{\partial%20f_{1}}{\partial%20x_{1}}}&\cdots%20&{\dfrac%20{\partial%20f_{1}}{\partial%20x_{n}}}\\\vdots%20&\ddots%20&\vdots%20\\{\dfrac%20{\partial%20f_{m}}{\partial%20x_{1}}}&\cdots%20&{\dfrac%20{\partial%20f_{m}}{\partial%20x_{n}}}\end{bmatrix}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \mathbf {J} ={\begin{bmatrix}{\dfrac {\partial \mathbf {f} }{\partial x_{1}}}&\cdots &{\dfrac {\partial \mathbf {f} }{\partial x_{n}}}\end{bmatrix}}={\begin{bmatrix}\nabla ^{\mathrm {T} }f_{1}\\\vdots \\\nabla ^{\mathrm {T} }f_{m}\end{bmatrix}}={\begin{bmatrix}{\dfrac {\partial f_{1}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{1}}{\partial x_{n}}}\\\vdots &\ddots &\vdots \\{\dfrac {\partial f_{m}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{m}}{\partial x_{n}}}\end{bmatrix}}}" />




# Hessian Matrix
The Hessian matrix 
<img  src="https://latex.codecogs.com/svg.latex?H"  alt="https://latex.codecogs.com/svg.latex?H" />  of f is a square <img  src="https://latex.codecogs.com/svg.latex?n%20\times%20n"  alt="https://latex.codecogs.com/svg.latex?n \times n" /> matrix, usually defined and arranged as follows:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\mathbf%20{H}%20_{f}={\begin{bmatrix}{\dfrac%20{\partial%20^{2}f}{\partial%20x_{1}^{2}}}&{\dfrac%20{\partial%20^{2}f}{\partial%20x_{1}\,\partial%20x_{2}}}&\cdots%20&{\dfrac%20{\partial%20^{2}f}{\partial%20x_{1}\,\partial%20x_{n}}}\\[2.2ex]{\dfrac%20{\partial%20^{2}f}{\partial%20x_{2}\,\partial%20x_{1}}}&{\dfrac%20{\partial%20^{2}f}{\partial%20x_{2}^{2}}}&\cdots%20&{\dfrac%20{\partial%20^{2}f}{\partial%20x_{2}\,\partial%20x_{n}}}\\[2.2ex]\vdots%20&\vdots%20&\ddots%20&\vdots%20\\[2.2ex]{\dfrac%20{\partial%20^{2}f}{\partial%20x_{n}\,\partial%20x_{1}}}&{\dfrac%20{\partial%20^{2}f}{\partial%20x_{n}\,\partial%20x_{2}}}&\cdots%20&{\dfrac%20{\partial%20^{2}f}{\partial%20x_{n}^{2}}}\end{bmatrix}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \mathbf {H} _{f}={\begin{bmatrix}{\dfrac {\partial ^{2}f}{\partial x_{1}^{2}}}&{\dfrac {\partial ^{2}f}{\partial x_{1}\,\partial x_{2}}}&\cdots &{\dfrac {\partial ^{2}f}{\partial x_{1}\,\partial x_{n}}}\\[2.2ex]{\dfrac {\partial ^{2}f}{\partial x_{2}\,\partial x_{1}}}&{\dfrac {\partial ^{2}f}{\partial x_{2}^{2}}}&\cdots &{\dfrac {\partial ^{2}f}{\partial x_{2}\,\partial x_{n}}}\\[2.2ex]\vdots &\vdots &\ddots &\vdots \\[2.2ex]{\dfrac {\partial ^{2}f}{\partial x_{n}\,\partial x_{1}}}&{\dfrac {\partial ^{2}f}{\partial x_{n}\,\partial x_{2}}}&\cdots &{\dfrac {\partial ^{2}f}{\partial x_{n}^{2}}}\end{bmatrix}}}" />


We can approximation Hessian Matrix by 
<img  src="https://latex.codecogs.com/svg.latex?H\approx%20J^TJ"  alt="https://latex.codecogs.com/svg.latex?H\approx J^TJ" />

Example: suppose you have the following function:

<img  src="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix}%20y_0%20=%2010\times(x_0+3)^2%20+%20(x_1-5)^2%20\\%20y_1%20=%20(x_0+1)\times%20x_1\\%20y_2%20=%20sin(x_0)\times%20x_1%20\end{matrix}\right."  alt="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix}
y_0 = 10\times(x_0+3)^2 + (x_1-5)^2 \\ 
y_1 = (x_0+1)\times x_1\\ 
y_2 = sin(x_0)\times x_1
\end{matrix}\right.
" />

In the next you will see how can we compute the Jacobian matrix with `Automatic Differentiation` and `Numerical Differentiation` 

Refs: [1](https://stats.stackexchange.com/questions/71154/when-an-analytical-jacobian-is-available-is-it-better-to-approximate-the-hessia)

# Automatic Differentiation
First let's create a generic functor,
```
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
```

Now lets prepare the input and out matrices and compute the jacobian:
```
Eigen::Matrix<double, 2, 1> x;
Eigen::Matrix<double, 3, 1> y;
Eigen::Matrix<double, 3,2> fjac;

Eigen::AutoDiffJacobian< VectorFunction<double,2, 3> > JacobianDerivatives;

// Set values in x, y and fjac...


x(0,0)=2.0;
x(1,0)=3.0;

JacobianDerivatives(x, &y, &fjac);

std::cout << "jacobian of matrix at "<< x(0,0)<<","<<x(1,0)  <<" is:\n " << fjac << std::endl;
```

Full source code [here](src/auto_diff.cpp)

# Numerical Differentiation

First let's create a generic functor,

```
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
```
Then inherit from the above generic functor and override the `operator()` and implement your function:

```
struct simpleFunctor : Functor<double>
{
    simpleFunctor(void): Functor<double>(2,3)
    {

    }
    // Implementation of the objective function
    // y0 = 10*(x0+3)^2 + (x1-5)^2
    // y1 = (x0+1)*x1
    // y2 = sin(x0)*x1
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        fvec(0) = 10.0*pow(x(0)+3.0,2) +pow(x(1)-5.0,2) ;
        fvec(1) = (x(0)+1)*x(1);
        fvec(2) = sin(x(0))*x(1);
        return 0;
    }
};
```
Now you can easily use it:
```

simpleFunctor functor;
Eigen::NumericalDiff<simpleFunctor,Eigen::NumericalDiffMode::Central> numDiff(functor);
Eigen::MatrixXd fjac(3,2);
Eigen::VectorXd x(2);
x(0) = 2.0;
x(1) = 3.0;
numDiff.df(x,fjac);
std::cout << "jacobian of matrix at "<< x(0)<<","<<x(1)  <<" is:\n " << fjac << std::endl;
```

Full source code: [here](src/numerical_diff.cpp)
    
    


