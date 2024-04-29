#include <Eigen/Dense>
#include <iostream>

// Transform, Translation, Scaling, Rotation2D and 3D rotations (Quaternion,
// AngleAxis)

#if KDL_FOUND == 1
#include <kdl/frames.hpp>
#endif

Eigen::Matrix3d eulerAnglesToRotationMatrix(double roll, double pitch,
                                            double yaw) {
  Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
  Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
  Eigen::Matrix3d rotationMatrix = q.matrix();
  return rotationMatrix;
}

void createRotationMatrix() {
  /*
  http://lavalle.pl/planning/node102.html
  http://euclideanspace.com/maths/geometry/rotations/conversions/index.htm

  Tait–Bryan angles: Z1Y2X3 in the wiki page:
  https://en.wikipedia.org/wiki/Euler_angles
    yaw:
        A yaw is a counterclockwise rotation of alpha about the  z-axis. The
    rotation matrix is given by

        R_z

        |cos(alpha) -sin(alpha) 0|
        |sin(apha)   cos(alpha) 0|
        |    0            0     1|

    pitch:
        R_y
        A pitch is a counterclockwise rotation of  beta about the  y-axis. The
    rotation matrix is given by

        |cos(beta)  0   sin(beta)|
        |0          1       0    |
        |-sin(beta) 0   cos(beta)|

    roll:
        A roll is a counterclockwise rotation of  gamma about the  x-axis. The
    rotation matrix is given by
        R_x
        |1          0           0|
        |0 cos(gamma) -sin(gamma)|
        |0 sin(gamma)  cos(gamma)|



        It is important to note that   R_z R_y R_x performs the roll first, then
  the pitch, and finally the yaw Roration matrix: R_z*R_y*R_x

  */
  double roll, pitch, yaw, alpha, beta, gamma;
  roll = M_PI / 4;
  pitch = M_PI / 3;
  yaw = M_PI / 6;

  alpha = yaw;
  beta = pitch;
  gamma = roll;

  Eigen::Matrix3d R_z, R_y, R_x;

  // yaw:
  R_z << cos(alpha), -sin(alpha), 0, sin(alpha), cos(alpha), 0, 0, 0, 1;

  // pitch:
  R_y << cos(beta), 0, sin(beta), 0, 1, 0, -sin(beta), 0, cos(beta);

  // roll:

  R_x << 1, 0, 0, 0, cos(gamma), -sin(gamma), 0, sin(gamma), cos(gamma);

  std::cout << eulerAnglesToRotationMatrix(roll, pitch, yaw) << std::endl;

  // It is important to note that   R_z R_y R_x performs the roll first, then
  // the pitch, and finally the yaw
  std::cout << R_z * R_y * R_x << std::endl;
}

void rotationMatrices() {

  //////////////////// Rotation Matrix (Tait–Bryan) ////////////////////////
  double roll, pitch, yaw;
  roll = M_PI / 2;
  pitch = M_PI / 2;
  yaw = M_PI / 6;
  std::cout << "Roll : " << roll << std::endl;
  std::cout << "Pitch : " << pitch << std::endl;
  std::cout << "Yaw : " << yaw << std::endl;

  // Roll, Pitch, Yaw to Rotation Matrix
  // Eigen::AngleAxis<double> rollAngle(roll, Eigen::Vector3d(1,0,0));
  Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

////////////////////////////////////////Comparing with
/// KDL////////////////////////////////////////
#if KDL_FOUND == 1
  KDL::Frame F;
  F.M = F.M.RPY(roll, pitch, yaw);
  std::cout << F.M(0, 0) << " " << F.M(0, 1) << " " << F.M(0, 2) << std::endl;
  std::cout << F.M(1, 0) << " " << F.M(1, 1) << " " << F.M(1, 2) << std::endl;
  std::cout << F.M(2, 0) << " " << F.M(2, 1) << " " << F.M(2, 2) << std::endl;

  double x, y, z, w;
  F.M.GetQuaternion(x, y, z, w);
  std::cout << "KDL Frame Quaternion:" << std::endl;
  std::cout << "x: " << x << std::endl;
  std::cout << "y: " << y << std::endl;
  std::cout << "z: " << z << std::endl;
  std::cout << "w: " << w << std::endl;
#endif

  Eigen::Matrix3d rotation;
  rotation = eulerAnglesToRotationMatrix(roll, pitch, yaw);

  double txLeft, tyLeft, tzLeft;
  txLeft = -1;
  tyLeft = 0.0;
  tzLeft = -4.0;

  Eigen::Affine3f t1;
  Eigen::Matrix4f M;
  Eigen::Vector3d translation;
  translation << txLeft, tyLeft, tzLeft;

  M << rotation(0, 0), rotation(0, 1), rotation(0, 2), translation(0, 0),
      rotation(1, 0), rotation(1, 1), rotation(1, 2), translation(1, 0),
      rotation(2, 0), rotation(2, 1), rotation(2, 2), translation(2, 0), 0, 0,
      0, 1;

  t1 = M;

  Eigen::Affine3d transform_2 = Eigen::Affine3d::Identity();

  // Define a translation of 2.5 meters on the x axis.
  transform_2.translation() << 2.5, 1.0, 0.5;

  // The same rotation matrix as before; tetha radians arround Z axis
  transform_2.rotate(yawAngle * pitchAngle * rollAngle);
  std::cout << transform_2.matrix() << std::endl;
  std::cout << transform_2.translation() << std::endl;
  std::cout << transform_2.translation().x() << std::endl;
  std::cout << transform_2.translation().y() << std::endl;
  std::cout << transform_2.translation().z() << std::endl;
}

void getingRollPitchYawFromRotationMatrix() {
  /*  http://planning.cs.uiuc.edu/node103.html

    |r11 r12 r13 |
    |r21 r22 r23 |
    |r31 r32 r33 |

    yaw: alpha=arctan(r21/r11)
    pitch: beta=arctan(-r31/sqrt( r32^2+r33^2 ) )
    roll: gamma=arctan(r32/r33)
*/
  double roll, pitch, yaw;
  roll = M_PI / 3;

  // these two will cause singularity
  // pitch = -M_PI / 2
  // pitch = M_PI / 2;
  pitch = M_PI / 4;
  yaw = M_PI / 6;
  Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

  Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
  Eigen::Matrix3d rotationMatrix = q.matrix();

  std::cout << "Rotation Matrix is:" << std::endl;
  std::cout << rotationMatrix << std::endl;

  std::cout << "roll is Pi/"
            << M_PI / atan2(rotationMatrix(2, 1), rotationMatrix(2, 2))
            << std::endl;
  std::cout << "pitch: Pi/"
            << M_PI / atan2(-rotationMatrix(2, 0),
                            std::pow(
                                rotationMatrix(2, 1) * rotationMatrix(2, 1) +
                                    rotationMatrix(2, 2) * rotationMatrix(2, 2),
                                0.5))
            << std::endl;

  std::cout << "yaw is Pi/"
            << M_PI / atan2(rotationMatrix(1, 0), rotationMatrix(0, 0))
            << std::endl;

  // Rotation Matrix to Tait–Bryan angles
  Eigen::Vector3d euler_angles = rotationMatrix.eulerAngles(2, 1, 0);

  std::cout << " Pi/"<<M_PI / euler_angles[0] << "," << " Pi/"<<M_PI / euler_angles[1] << ","
            << " Pi/"<<M_PI / euler_angles[2] << std::endl;

}
int main() { getingRollPitchYawFromRotationMatrix(); }
