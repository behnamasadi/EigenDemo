#include <iostream>
#include <vector>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

/*

https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
https://en.wikipedia.org/wiki/Non-linear_least_squares
https://lingojam.com/SuperscriptGenerator
https://lingojam.com/SubscriptGenerator

consider m data points:
(x₁,y₁),(x₂,y₂),...(xₘ,yₘ)

and a functions which has n parameters:
β=(β₁, β₂,..., βₙ)

where m>=n

this gives us m residuals:
rᵢ=yᵢ-f(xᵢ,β)

our objective is minimize the:

S=Σ(rᵢ)²

The minimum value of S occurs when the gradient is zero

Lets say we have the following dataset:

i	    1	    2	    3	    4	    5	    6	    7
x	    0.038	0.194	0.425	0.626	1.253	2.500	3.740
y    	0.050	0.127	0.094	0.2122	0.2729	0.2665	0.3317

and we have the folliwng function:

y=(β₁*x)/(β₂+x)

so our r is
r₁=y₁ - (β₁*x₁)/(β₂+x₁)
r₂=y₂ - (β₁*x₂)/(β₂+x₂)
r₃=y₃ - (β₁*x₃)/(β₂+x₃)
r₄=y₄ - (β₁*x₄)/(β₂+x₄)
r₅=y₅ - (β₁*x₅)/(β₂+x₅)
r₆=y₆ - (β₁*x₆)/(β₂+x₆)
r₇=y₇ - (β₁*x₇)/(β₂+x₇)

r(β)₂ₓ₇  -> J₇ₓ₂


σrᵢ/σβ₁=-xᵢ/(β₂+xᵢ)

σrᵢ/σβ₂=(β₁*xᵢ)/(β₂xᵢ)²


βᵏ⁺¹=βᵏ- (JᵀJ)⁻¹Jᵀr(βᵏ)
*/



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

typedef std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > Point2DVector;

//Point2DVector GeneratePoints(const unsigned int numberOfPoints)

struct SubstrateConcentrationFunctor : Functor<double>
{

    SubstrateConcentrationFunctor(Eigen::MatrixXd points): Functor<double>(points.cols(),points.rows())
    {
        this->Points = points;
    }

    int operator()(const Eigen::VectorXd &z, Eigen::VectorXd &r) const
    {
        double x_i,y_i,beta1,beta2;
        for(unsigned int i = 0; i < this->Points.rows(); ++i)
        {
            y_i=this->Points.row(i)(1);
            x_i=this->Points.row(i)(0);
            beta1=z(0);
            beta2=z(1);
            r(i) =y_i-(beta1*x_i) /(beta2+x_i);
        }

        return 0;
    }
    Eigen::MatrixXd Points;

    int inputs() const { return 2; } // There are two parameters of the model, beta1, beta2
    int values() const { return this->Points.rows(); } // The number of observations
};



void SubstrateConcentrationLeastSquare()
{
/*
our data:

x     y
0.038,0.050;
0.194,0.127;
0.425,0.094;
0.626,0.2122;
1.253,0.2729;
2.500,0.2665;
3.740,0.3317;

the last column in the matrix should be "y"

*/


    Eigen::MatrixXd points(7,2);


    points.row(0)<< 0.038,0.050;
    points.row(1)<<0.194,0.127;
    points.row(2)<<0.425,0.094;
    points.row(3)<<0.626,0.2122;
    points.row(4)<<1.253,0.2729;
    points.row(5)<<2.500,0.2665;
    points.row(6)<<3.740,0.3317;

    SubstrateConcentrationFunctor functor(points);
    Eigen::NumericalDiff<SubstrateConcentrationFunctor,Eigen::NumericalDiffMode::Central> numDiff(functor);

    std::cout<<"functor.Points.size(): " <<functor.Points.size()<<std::endl;
    Eigen::VectorXd beta(2);
    beta<<0.9,0.2;

    Eigen::MatrixXd J(7,2);
    numDiff.df(beta,J);

    std::cout << "jacobian of matrix at "<< beta(0)<<","<<beta(1)  <<" is:\n " << J << std::endl;


    Eigen::VectorXd r(7);
    functor(beta,r);
    std::cout<<r<<std::endl;

    //βᵏ⁺¹=βᵏ- (JᵀJ)⁻¹Jᵀr(βᵏ)
    for(int i=0;i<10;i++)
    {
        numDiff.df(beta,J);
        std::cout<<"J: \n" << J<<std::endl;
        functor(beta,r);
        std::cout<<"r: \n" << r<<std::endl;
        beta=beta-(J.transpose()*J).inverse()*J.transpose()*r ;


    }
    std::cout<<"beta: \n" << beta<<std::endl;

    //optimal beta 0.36,0.556;

}




int main()
{
    SubstrateConcentrationLeastSquare();
}

