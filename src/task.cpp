#include <math.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/NumericalDiff>


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
    link_offset.translation()(1) = 1.;
    //link_offset.translation()(0) = 1.;

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

    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

    std::cout<<"cond: "<<cond <<std::endl;

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
    numericalDifferentiationFKFunctor(): Functor<double>(3,3) {}

    // Implementation of the objective function
    int operator()(const Eigen::VectorXd &q, Eigen::VectorXd &fvec) const
    {
        trafo2d_t t= forward_kinematics(q);
        double theta=atan2( t.rotation()(1,0),t.rotation()(0,0));
        double x = t.translation()(0);
        double y = t.translation()(1);

        fvec(0) = x;
        fvec(1) = y;
        fvec(2) = theta;

        return 0;
    }
};

Eigen::MatrixXd numericalDifferentiationFK(const Eigen::VectorXd &q)
{
    numericalDifferentiationFKFunctor functor;
    Eigen::NumericalDiff<numericalDifferentiationFKFunctor> numDiff(functor);
    Eigen::MatrixXd fjac(3,3);
    numDiff.df(q,fjac);
    return fjac;
}

Eigen::VectorXd transformationMatrixToPose(trafo2d_t const &m)
{
    double theta=atan2( m.rotation()(1,0),m.rotation()(0,0));
    double x = m.translation()(0);
    double y = m.translation()(1);

    Eigen::VectorXd fvec(3);
    fvec(0) = x;
    fvec(1) = y;
    fvec(2) = theta;
    return fvec;
}

Eigen::VectorXd distanceError(trafo2d_t const &golesStart, trafo2d_t const &poseStart)
{
     return transformationMatrixToPose(golesStart)-transformationMatrixToPose(poseStart);
}

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
    q=remainder(q,sign*2*M_PI);
    if(-2*M_PI<=q && q <=-M_PI)
        q=q+2*M_PI;

    else if(+M_PI<=q && q <=2*M_PI)
        q=2*M_PI-q;
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
vector_t inverse_kinematics(vector_t const & q_start, trafo2d_t const & goal )
{
    vector_t q=q_start;
    vector_t delta_q(3);


    double epsilon=1e-12;

    int i=0;

    while( (distanceError(goal,forward_kinematics(q)).squaredNorm()>epsilon)  && (i<200000000)  )
    {
        Eigen::MatrixXd jacobian=numericalDifferentiationFK(q);

        Eigen::MatrixXd j_pinv=pseudoInverse(jacobian);

        std::cout<<"determinant jacobian:" <<jacobian.determinant()<<std::endl;


        std::cout<<"jacobian:\n" <<jacobian<<std::endl;

        std::cout<<"determinant j_pinv:\n" <<j_pinv.determinant()<<std::endl;
        std::cout<<"jacobian*jacobian.transpose:" <<(jacobian*jacobian.transpose()).determinant()<<std::endl;

        std::cout<<"jacobian*jacobian.transpose:" <<(j_pinv*jacobian).determinant()<<std::endl;



        std::cout<<"error:" <<distanceError(goal,forward_kinematics(q)).squaredNorm()<<std::endl;
        //A matrix is singular if and only if its determinant is zero.

        //When a robot is controlled in Cartesian space and passes near a singularity, the velocity of some joints becomes suddenly very high.


        Eigen::VectorXd delta_p=transformationMatrixToPose(goal)-transformationMatrixToPose(forward_kinematics(q) );


        delta_p(0)=delta_p(0)/100;
        delta_p(1)=delta_p(1)/100;
        //delta_p(2)=delta_p(1)/errorValue*step;




        delta_q=j_pinv*delta_p;
        q=q+delta_q;



        normaliseAngle2(q);

        std::cout<<"q:" <<q<<std::endl;
        i++;
    }
    return q;
}

vector_t inverse_kinematics(trafo2d_t const & goal )
{

}


/**
 * An example how the inverse kinematics can be used.
 * It should not be required to change this code.
 */





int main()
{
    vector_t q_start(3);
    q_start.setConstant(-0.1);

    trafo2d_t goal = trafo2d_t::Identity();
    goal.translation()(0) = 1.0;
    //goal.translation()(0) = 0.0;

    Eigen::VectorXd goalJointValues(3);
//    goalJointValues(0) = M_PI;
//    goalJointValues(1) = M_PI/2;
//    goalJointValues(2) = M_PI/2;



    goalJointValues(0) = -M_PI/2;
    goalJointValues(1) = -M_PI/2;
    goalJointValues(2) = -M_PI;

//    goalJointValues(0) = 0;
//    goalJointValues(1) = 0;
//    goalJointValues(2) = 0;

    goal = forward_kinematics(goalJointValues);

    std::cout <<"goal:\n" <<transformationMatrixToPose(goal) << std::endl;



    vector_t result = inverse_kinematics(q_start,goal);
    std::cout <<"joint values for the given pose are:\n" <<result << std::endl;
    std::cout << "goal pose is:\n" << transformationMatrixToPose(goal) <<std::endl;
    std::cout << "estimated pose from IK is:\n" << transformationMatrixToPose(forward_kinematics(result)) <<std::endl;

    return 0;
}

//https://se.mathworks.com/matlabcentral/fileexchange/61380-3dof-inverse-kinematics-pseudoinverse-jacobian
//https://stackoverflow.com/questions/10115354/inverse-kinematics-with-opengl-eigen3-unstable-jacobian-pseudoinverse
