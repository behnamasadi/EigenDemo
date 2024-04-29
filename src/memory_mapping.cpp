#include <iostream>
#include <Eigen/Dense>
#include <vector>


struct point {
    double a;
    double b;
};

void eigenMapExample()
{
    ////////////////////////////////////////First Example/////////////////////////////////////////
    Eigen::VectorXd solutionVec(12,1);
    solutionVec<<1,2,3,4,5,6,7,8,9,10,11,12;
    Eigen::Map<Eigen::MatrixXd> solutionColMajor(solutionVec.data(),4,3);

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> >solutionRowMajor (solutionVec.data());


    std::cout << "solutionColMajor: "<< std::endl;
    std::cout << solutionColMajor<< std::endl;

    std::cout << "solutionRowMajor"<< std::endl;
    std::cout << solutionRowMajor<< std::endl;

    ////////////////////////////////////////Second Example/////////////////////////////////////////

    // https://stackoverflow.com/questions/49813340/stdvectoreigenvector3d-to-eigenmatrixxd-eigen

    int array[9];
    for (int i = 0; i < 9; ++i) {
        array[i] = i;
    }

    Eigen::MatrixXi a(9, 1);
    a = Eigen::Map<Eigen::Matrix3i>(array);
    std::cout << a << std::endl;

    std::vector<point> pointsVec;
    point point1, point2, point3;

    point1.a = 1.0;
    point1.b = 1.5;

    point2.a = 2.4;
    point2.b = 3.5;

    point3.a = -1.3;
    point3.b = 2.4;

    pointsVec.push_back(point1);
    pointsVec.push_back(point2);
    pointsVec.push_back(point3);

    Eigen::Matrix2Xd pointsMatrix2d = Eigen::Map<Eigen::Matrix2Xd>(
        reinterpret_cast<double*>(pointsVec.data()), 2,  long(pointsVec.size()));

    Eigen::MatrixXd pointsMatrixXd = Eigen::Map<Eigen::MatrixXd>(
        reinterpret_cast<double*>(pointsVec.data()), 2, long(pointsVec.size()));

    std::cout << pointsMatrix2d << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << pointsMatrixXd << std::endl;
    std::cout << "==============================" << std::endl;

    std::vector<Eigen::Vector3d> eigenPointsVec;
    eigenPointsVec.push_back(Eigen::Vector3d(2, 4, 1));
    eigenPointsVec.push_back(Eigen::Vector3d(7, 3, 9));
    eigenPointsVec.push_back(Eigen::Vector3d(6, 1, -1));
    eigenPointsVec.push_back(Eigen::Vector3d(-6, 9, 8));

    Eigen::MatrixXd pointsMatrix = Eigen::Map<Eigen::MatrixXd>(eigenPointsVec[0].data(), 3, long(eigenPointsVec.size()));

    std::cout << pointsMatrix << std::endl;
    std::cout << "==============================" << std::endl;

    pointsMatrix = Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double*>(eigenPointsVec.data()), 3, long(eigenPointsVec.size()));

    std::cout << pointsMatrix << std::endl;

    std::vector<double> aa = { 1, 2, 3, 4 };
    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(aa.data(), long(aa.size()));
}

int main()
{
    eigenMapExample();
}
