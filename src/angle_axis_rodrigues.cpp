#include <iostream>
#include <Eigen/Dense>
#include <iostream>

Eigen::Matrix3d eulerAnglesToRotationMatrix(double roll, double pitch,
                                            double yaw) {
  Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
  Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
  Eigen::Matrix3d rotationMatrix = q.matrix();
  return rotationMatrix;
}


int main()
{
    //////////////////////////////////////// create angle axis (rodrigues) ///////////////////////////////////////////


    Eigen::Vector3d vector3d(2.3,3.1,1.7);
    Eigen::Vector3d vector3dNormalized=vector3d.normalized();
    double theta=M_PI/7;
    Eigen::AngleAxisd angleAxisConversion(theta,vector3dNormalized);


    //////////////////////////////////////// angle axis (rodrigues) to rotation matrix ///////////////////////////////////////////
    Eigen::Matrix3d rotationMatrixConversion;
    rotationMatrixConversion=angleAxisConversion.toRotationMatrix();



    //////////////////////////////////////// rotation matrix to angle axis (rodrigues) ///////////////////////////////////////////


    Eigen::Matrix3d rotationMatrix;

    double roll, pitch, yaw;
    roll=M_PI/2;
    pitch=M_PI/2;
    yaw=M_PI/6;

    rotationMatrix= eulerAnglesToRotationMatrix(roll, pitch,yaw);

    Eigen::AngleAxisd rodrigues(rotationMatrix );
    std::cout<<"Rodrigues Angle:\n"<<rodrigues.angle() <<std::endl;

    std::cout<<"Rodrigues Axis:" <<std::endl;

    std::cout<<rodrigues.axis().x() <<std::endl;
    std::cout<<rodrigues.axis().y() <<std::endl;
    std::cout<<rodrigues.axis().z() <<std::endl;



}
