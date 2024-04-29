#include <iostream>
#include <Eigen/Dense>

void gramSchmidtOrthogonalization(Eigen::MatrixXd &matrix,Eigen::MatrixXd &orthonormalMatrix)
{
/*
    In this method you make every column perpendicular to it's previous columns,
    here if a and b are representation vector of two columns, c=b-((b.a)/|a|).a
        ^
       /
    b /
     /
    /
    ---------->
        a
        ^
       /|
    b / |
     /  | c
    /   |
    ---------->
        a
    you just have to normilze every vector after make it perpendicular to previous columns
    so:
    q1=a.normalized();
    q2=b-(b.q1).q1
    q2=q2.normalized();
    q3=c-(c.q1).q1 - (c.q2).q2
    q3=q3.normalized();
    Now we have Q, but we want A=QR so we just multiply both side by Q.transpose(), since Q is orthonormal, Q*Q.transpose() is I
    A=QR;
    Q.transpose()*A=R;
*/
    Eigen::VectorXd col;
    for(int i=0;i<matrix.cols();i++)
    {
        col=matrix.col(i);
        col=col.normalized();
        for(int j=0;j<i-1;j++)
        {
            //orthonormalMatrix.col(i)
        }

        orthonormalMatrix.col(i)=col;
    }
    Eigen::MatrixXd A(4,3);

    A<<1,2,3,-1,1,1,1,1,1,1,1,1;
    Eigen::Vector4d a=A.col(0);
    Eigen::Vector4d b=A.col(1);
    Eigen::Vector4d c=A.col(2);

    Eigen::Vector4d q1=  a.normalized();
    Eigen::Vector4d q2=b-(b.dot(q1))*q1;
    q2=q2.normalized();

    Eigen::Vector4d q3=c-(c.dot(q1))*q1 - (c.dot(q2))*q2;
    q3=q3.normalized();

    std::cout<< "q1:"<<std::endl;
    std::cout<< q1<<std::endl;
    std::cout<< "q2"<<std::endl;
    std::cout<< q2<<std::endl;
    std::cout<< "q3:"<<std::endl;
    std::cout<< q3<<std::endl;

    Eigen::MatrixXd Q(4,3);
    Q.col(0)=q1;
    Q.col(1)=q2;
    Q.col(2)=q3;

    Eigen::MatrixXd R(3,3);
    R=Q.transpose()*(A);


    std::cout<<"Q"<<std::endl;
    std::cout<< Q<<std::endl;


    std::cout<<"R"<<std::endl;
    std::cout<< R.unaryExpr(std::ptr_fun(exp))<<std::endl;



    //MatrixXd A(4,3), thinQ(4,3), Q(4,4);

    Eigen::MatrixXd thinQ(4,3), q(4,4);

    //A.setRandom();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    q = qr.householderQ();
    thinQ.setIdentity();
    thinQ = qr.householderQ() * thinQ;
    std::cout << "Q computed by Eigen" << "\n\n" << thinQ << "\n\n";
    std::cout << q << "\n\n" << thinQ << "\n\n";


}

void gramSchmidtOrthogonalizationExample()
{
    Eigen::MatrixXd matrix(3,4),orthonormalMatrix(3,4) ;
    matrix=Eigen::MatrixXd::Random(3,4);////A.setRandom();


    gramSchmidtOrthogonalization(matrix,orthonormalMatrix);
}



int main()
{

}
