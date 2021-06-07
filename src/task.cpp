#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <math.h>


#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>


typedef Eigen::VectorXd vector_t;
typedef Eigen::MatrixXd matrix_t;
typedef Eigen::Transform<double,2,Eigen::Affine> trafo2d_t;

/**************************************************
 * A function to compute the forward kinematics of
 * a planar 3-link robot.
 *************************************************/
trafo2d_t forward_kinematics(vector_t const & q ) {
    // check that the joint angle vector has the correct size
    assert( q.size() == 3 );

    // define a constant offset between two joints
    trafo2d_t link_offset = trafo2d_t::Identity();
    //link_offset.translation()(1) = 1.;
    link_offset.translation()(0) = 1.;

    // define the start pose
    trafo2d_t trafo = trafo2d_t::Identity();
    //std::cout<<"trafo.matrix()\n" <<trafo.matrix()<<std::endl;

    for(int joint_idx = 0; joint_idx < 3 ; joint_idx++ ) {
        // add the rotation contributed by this joint
        //std::cout<<Eigen::Rotation2D<double>(q(joint_idx)).matrix() <<std::endl;


        trafo *= Eigen::Rotation2D<double>(q(joint_idx));
        // add the link offset to this position
        //std::cout<<link_offset.matrix() <<std::endl;


        trafo = trafo * link_offset;
    }
    return trafo;
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

struct numericalDifferentiationFKFunctor : Functor<double>
{
    // Simple constructor
    numericalDifferentiationFKFunctor(): Functor<double>(3,2) {}

    // Implementation of the objective function
    int operator()(const Eigen::VectorXd &z, Eigen::VectorXd &fvec) const
    {
        trafo2d_t t= forward_kinematics(z);
        double theta=atan2( t.rotation()(1,0),t.rotation()(0,0));
        double x = t.translation()(0);
        double y = t.translation()(1);

        fvec(0) = x;
        fvec(1) = y;
        //fvec(2) = theta;

        return 0;
    }
};



Eigen::VectorXd transformatioToPose(trafo2d_t const &m)
{
    double theta=atan2( m.rotation()(1,0),m.rotation()(0,0));
    double x = m.translation()(0);
    double y = m.translation()(1);

    Eigen::VectorXd fvec(2);
    fvec(0) = x;
    fvec(1) = y;
    //fvec(2) = theta;
    return fvec;
}

Eigen::VectorXd p(const Eigen::VectorXd &q)
{
    trafo2d_t t= forward_kinematics(q);
    double theta=atan2( t.rotation()(1,0),t.rotation()(0,0));
    double x = t.translation()(0);
    double y = t.translation()(1);

    Eigen::VectorXd fvec(2);
    fvec(0) = x;
    fvec(1) = y;
    //fvec(2) = theta;
    return fvec;
}

Eigen::MatrixXd numericalDifferentiationFK(const Eigen::VectorXd &q)
{

    numericalDifferentiationFKFunctor functor;
    Eigen::NumericalDiff<numericalDifferentiationFKFunctor> numDiff(functor);



    //std::cout << "starting q: \n" << q << std::endl;

    Eigen::MatrixXd fjac(2,3);
    numDiff.df(q,fjac);

    //std::cout << "numerical differentiation at \n"<<q <<"\n is: \n" << fjac << std::endl;

    return fjac;
}


/*************************************************
 * Task:
 * Complete this inverse kinematics function for the robot defined by
 * the forward kinematics function defined above.
 * It should return the joint angles q for the given goal specified in the
 * corresponding parameter.
 * Only the translation (not rotation) part of the goal has to match.
 * 
 *
 * Hints:
 * - This is an non-linear optimization problem which can be solved by using
 *   an iterative algorithm.
 * - To obtain the jacobian, use numerical differentiation
 * - To invert the jacobian use Eigen::JacobiSVD
 * - The algorithm should stop when norm of the error is smaller than 1e-3
 * - The algorithm should also stop when 200 iterations are reached
 ************************************************/
vector_t inverse_kinematics(vector_t const & q_start, trafo2d_t const & goal ) {
}

vector_t inverse_kinematics(trafo2d_t const & goal )
{
}


/**
 * An example how the inverse kinematics can be used.
 * It should not be required to change this code.
 */



using namespace Eigen;
using std::cout;

template <typename T> inline constexpr
int signum(T x, std::false_type is_signed) {
    return T(0) < x;
}

template <typename T> inline constexpr
int signum(T x, std::true_type is_signed) {
    return (T(0) < x) - (x < T(0));
}

template <typename T> inline constexpr
int signum(T x) {
    return signum(x, std::is_signed<T>());
}

template <typename  T>
void normaliseAngle(T &q)
{
    int sign=signum(q);
    q=fabs(q);
    q=sign*remainder(q,2*M_PI);
    if(sign<0)
        q=q+2*M_PI;
}

template <typename  T>
void normaliseAngle2(T &q)
{
    int sign=signum(q);
    //q=fabs(q);
    q=remainder(q,sign*2*M_PI);
    if(-2*M_PI<=q && q <=-M_PI)
    {
            q=q+2*M_PI;
    }

    else if(+M_PI<=q && q <=2*M_PI)
    {
            q=2*M_PI-q;
    }

}



void normaliseAngle(Eigen::VectorXd &q)
{
    for(std::size_t i=0;i<q.rows();i++)
    {
        normaliseAngle(q(i));
    }
}

void normaliseAngle2(Eigen::VectorXd &q)
{
    for(std::size_t i=0;i<q.rows();i++)
    {
        normaliseAngle2(q(i));
    }
}


int main()
{
//    vector_t q_start(3);
//    q_start.setConstant(-0.1);

//    trafo2d_t goal = trafo2d_t::Identity();
//    goal.translation()(0) = 1.;

//    vector_t q(3);
//    q<<-M_PI/2,M_PI/2,M_PI/3;
//    trafo2d_t t= forward_kinematics(q);
//    std::cout<<"forward_kinematics trafo2d_t: \n"  << t.matrix() <<std::endl;
//    std::cout<<"t.translation(): \n"  << t.translation()(0) << ","<<t.translation()(1) <<std::endl;
//    std::cout<<"rotation angle: \n"  << atan2( t.rotation()(1,0),t.rotation()(0,0))/M_PI <<std::endl;


    {
        Eigen::VectorXd q(3);
        q(0) = M_PI/2;
        q(1) = -M_PI;
        q(2) = M_PI/4;

//        q(0) = 0;
//        q(1) = 0;
//        q(2) = 0;

        trafo2d_t goal = forward_kinematics(q);

        std::cout << "goal is: "<< transformatioToPose(goal)<<std::endl;

//        q(0) =q(0)+ 0.02;
//        q(1) =q(1)+ 0.02;
//        q(2) =q(2)+ 0.02;


        q(0) = 0;
        q(1) = 0;
        q(2) = 0;



        vector_t delta_q(3);

        delta_q(0) = 0.01;
        delta_q(1) = 0.01;
        delta_q(2) = 0.01;
        double epsilon=0.001;

        int i=0;

        //p(q+delta_q)-transformatioToPose(goal).array().abs().sum()

        //std::cout<<"initial error: \n" <<(p(q)-transformatioToPose(goal)).array().abs().sum() <<std::endl;

        std::cout<<"initial error: \n" <<p(q)-transformatioToPose(goal) <<std::endl;

        while( ((p(q)-transformatioToPose(goal)).array().abs().sum() >epsilon ) && (i<20000))
        {
            double errorValue=(p(q)-transformatioToPose(goal)).transpose()*(p(q)-transformatioToPose(goal));
            std::cout<<"error value" <<errorValue<<std::endl;
            Eigen::MatrixXd jacobian=numericalDifferentiationFK(q);
            Eigen::MatrixXd j_pinv=pseudoInverse(jacobian);
//            std::cout<<"jacobian: \n" <<jacobian <<std::endl;
//            std::cout<<"j_pinv: \n" <<j_pinv <<std::endl;
            //std::cout<<"J^T*J: " <<(j_pinv*jacobian).array().abs().sum() <<std::endl;
//            std::cout<<"J^T*J: " <<j_pinv*jacobian <<std::endl;
            //Eigen::VectorXd delta_p=p(q+delta_q)-p(q);

            Eigen::VectorXd delta_p=transformatioToPose(goal)-p(q);
            double step=10;
            if(errorValue>1)
            {
                delta_p(0)=delta_p(0)/errorValue*step;
                delta_p(1)=delta_p(1)/errorValue*step;
            }
            else
            {
                delta_p(0)=delta_p(0)/step;
                delta_p(1)=delta_p(1)/step;

            }

            //std::cout<<"error: \n" <<(transformatioToPose(goal)-p(q)) <<std::endl;

            delta_q=j_pinv*delta_p;

            q=q+delta_q;
            normaliseAngle2(q);

            i++;
            std::cout<<"i: \n" <<i<<std::endl;
            //std::cout<<"delta_q: \n" <<delta_q<<std::endl;
            std::cout<<"q: \n" <<q<<std::endl;
            //std::cout<<"diff is: \n" <<(p(q+delta_q)-transformatioToPose(goal)) <<std::endl;
        }
//        std::cout<<"q: \n" <<q <<std::endl;
//        std::cout<<"q(0): \n" <<remainder(q(0),2*M_PI) <<std::endl;
//        std::cout<<"q(1): \n" <<remainder(q(1),2*M_PI) <<std::endl;
//        std::cout<<"q(2): \n" <<remainder(q(1),2*M_PI) <<std::endl;
        std::cout<<"transformatioToPose(goal): \n" <<transformatioToPose(goal) <<std::endl;
        std::cout<<"p(q): \n" <<p(q) <<std::endl;
    }
    double t=-5*M_PI/2;
    normaliseAngle2(t);
    std::cout<<"t: " <<t <<std::endl;
    t=+5*M_PI/2;
    normaliseAngle2(t);
    std::cout<<"t: " <<t <<std::endl;
    return 0;
}

//https://se.mathworks.com/matlabcentral/fileexchange/61380-3dof-inverse-kinematics-pseudoinverse-jacobian
//https://stackoverflow.com/questions/10115354/inverse-kinematics-with-opengl-eigen3-unstable-jacobian-pseudoinverse
