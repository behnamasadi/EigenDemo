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

Eigen::Vector3d rotationMatrixToEulerAngles(Eigen::Matrix3d rotationMatrix)
{
    Eigen::Vector3d euler_angles = rotationMatrix.eulerAngles(2, 1, 0);
    return euler_angles;
}

int main()
{

    double roll, pitch, yaw;

    roll = M_PI / 8;
    pitch = M_PI / 4;
    yaw = M_PI / 9;


    std::cout << "set angles:" << std::endl;

    std::cout << "roll: Pi/" << M_PI / roll << " ,pitch: Pi/" << M_PI / pitch << " ,yaw : Pi/" << M_PI / yaw
              << std::endl;

    Eigen::Matrix3d rotationMatrix = eulerAnglesToRotationMatrix(roll, pitch, yaw);
    Eigen::Vector3d euler_angles = rotationMatrixToEulerAngles(rotationMatrix);

    std::cout << "retrieved angles:" << std::endl;

    std::cout << "roll: Pi/" << M_PI / euler_angles[2] << " ,pitch: Pi/" << M_PI / euler_angles[1] << " ,yaw : Pi/"
              << M_PI / euler_angles[0] << std::endl;

}
