#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

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

void SVD_Example()
{
    /*

     AX=0;
     A, U, V=SVD(A);
     A* U(Index of last column)=0;
     */

    Eigen::MatrixXd A;
    A.setRandom(4,3);
    std::cout<<"Matrix A" <<std::endl;
    std::cout<<A <<std::endl;
    //Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);




    std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;

    //std::cout << "Its singular values are:" << std::endl << svd.singularValues().asDiagonal() << std::endl;




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
    std::cout<<U*Sigma*V.transpose() <<std::endl;


    std::cout<<"This should be very close to zero: A - U*Sigma*V.transpose()" <<std::endl;
    std::cout<<A -U*Sigma*V.transpose() <<std::endl;


    std::cout<<"This should be zero vector (solution of the problem A*V.col( V.cols()-1))" <<std::endl;
    std::cout<<A*V.col( V.cols()-1)<<std::endl;

    //std::cout<<"pinv of A=V*Sigma.pinv*U.transpose()=V*Sigma.transpose()*U.transpose()" <<std::endl;
    std::cout<<"pinv of A=V*Sigma.inverse()*U.transpose()" <<std::endl;

    Eigen::MatrixXd A_pinv= V*Sigma.inverse()*U.transpose();
    std::cout<<A_pinv <<std::endl;


    std::cout<<((A.transpose()*A).inverse() *A.transpose())<<std::endl;


//    std::cout<<"A_pinv*A=I" <<std::endl;
//    std::cout<<A_pinv*A <<std::endl;
//    std::cout<<((A.transpose()*A).inverse() *A.transpose())*A <<std::endl;





}

/**
 * An example how the inverse kinematics can be used.
 * It should not be required to change this code.
 */

using namespace Eigen;
using std::cout;

int main()
{
    vector_t q_start(3);
    q_start.setConstant(-0.1);

    trafo2d_t goal = trafo2d_t::Identity();
    goal.translation()(0) = 1.;

    //std::cout<< goal.matrix()<<std::endl;
    //std::cout<< q_start.matrix() <<std::endl;



    //vector_t result = inverse_kinematics(q_start,goal);
    //std::cout << result << std::endl;

    vector_t q(3);
    //q<<M_PI/4,M_PI/2,M_PI/3;
    //q<<0,0,0;
    //q<<-M_PI/2,0,0;
    q<<-M_PI/2,M_PI/2,0;
    //std::cout<< q <<std::endl;



    trafo2d_t t= forward_kinematics(q);

    std::cout<<  t.matrix() <<std::endl;


    vector_t pose(3);
    //pose.setConstant(-0.1);
    //pose.setConstant(0);
    pose<<1,2,1;
    //std::cout<<"pose\n" << pose <<std::endl;
    //std::cout<<t.matrix() <<std::endl;
    std::cout<< (t.matrix()* pose)  /pose(2)  <<std::endl;
    //std::cout<<  pose.homogeneous()  <<std::endl;



    //std::cout<<"pose(1)\n"<< pose(2) <<std::endl;

    //std::cout<< pose<<std::endl;

    //SVD_Example();


    Eigen::Matrix< double, 3, 1> v ;
    v << 1, 2, 3;
    Eigen::Matrix< double, 3, 3> m = v.array().sqrt().matrix().asDiagonal();

    std::cout << m<< "\n";


    std::cout << m.transpose() << "\n";

    return 0;
}
