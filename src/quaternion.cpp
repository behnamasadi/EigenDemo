#include <Eigen/Dense>
#include <iostream>

// very good refrence:
// https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html

Eigen::Matrix3d eulerAnglesToRotationMatrix(double roll, double pitch,
                                            double yaw) {
  Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
  Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
  Eigen::Matrix3d rotationMatrix = q.matrix();
  return rotationMatrix;
}

void quaternionFromRollPitchYaw() {

  //////////////// rotation matrix to quaternion   ////////////////

  double roll, pitch, yaw;

  roll = M_PI / 4;
  pitch = M_PI / 3;
  yaw = M_PI / 6;

  Eigen::Matrix3d rotationMatrix =
      eulerAnglesToRotationMatrix(roll, pitch, yaw);

  Eigen::Quaterniond quaternionFromRotationMatrix(rotationMatrix);

  std::cout << "Quaternion X: " << quaternionFromRotationMatrix.x()
            << std::endl;
  std::cout << "Quaternion Y: " << quaternionFromRotationMatrix.y()
            << std::endl;
  std::cout << "Quaternion Z: " << quaternionFromRotationMatrix.z()
            << std::endl;
  std::cout << "Quaternion W: " << quaternionFromRotationMatrix.w()
            << std::endl;

  //  // Rotation Matrix to Euler Angle (Proper)
  //  Eigen::Vector3d euler_angles = rotationMatrix.eulerAngles(2, 0, 2);

  //  // Eigen::Quaterniond
  //  Eigen::Quaterniond tmp =
  //      Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitZ()) *
  //      Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitX()) *
  //      Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitZ());

  //  Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;

  //////////////// Quaternion to Rotation Matrix ////////////////
  Eigen::Matrix3d rotationMatrixFromQuaternion =
      quaternionFromRotationMatrix.matrix();
  std::cout << "3x3 Rotation Matrix\n"
            << rotationMatrixFromQuaternion << std::endl;
}

struct Quaternion {
  double w, x, y, z;
};

struct Point {
  double x, y, z;
};

Quaternion quaternionMultiplication(Quaternion p, Quaternion q) {
  //(t0, t1, t2, t3) = (a1, b1, c1, d1) ✕ (a2, b2, c2, d2)

  // p=(a1 + b1*i, c1*j, d1*k)=(w,x,y,z)
  // q=(a2 + b2*i, c2*j, d2*k)=(w,x,y,z)
  double a1, b1, c1, d1, a2, b2, c2, d2;

  a1 = p.w;
  b1 = p.x;
  c1 = p.y;
  d1 = p.z;

  a2 = q.w;
  b2 = q.x;
  c2 = q.y;
  d2 = q.z;

  Quaternion t;

  // t0 = (a1a2 − b1b2 − c1c2 − d1d2)
  t.w = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2;

  // t1 = (a1b2+b1a2+c1d2 -d1c2 )
  t.x = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2;

  // t2 = (a1c2 -b1d2 +c1a2+ d1b2 )
  t.y = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2;

  // t3 = (a1d2+ b1c2 -c1b2+ d1a2 )
  t.z = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2;

  return t;
}

Quaternion quaternionInversion(Quaternion q) {
  // The inverse of a quaternion is obtained by negating the imaginary
  // components:

  q.x = -q.x;
  q.y = -q.y;
  q.z = -q.z;
  return q;
}

Point QuaternionRotation(Quaternion q, Point p) {
  // Step 1:Convert the point to be rotated into a quaternion
  // p = (p0, p1, p2, p3) = ( 0, x, y, z )
  Quaternion tmp;
  tmp.w = 0;
  tmp.x = p.x;
  tmp.y = p.y;
  tmp.z = p.z;

  // Step 2: Perform active rotation: when the point is rotated with respect to
  // the coordinate system p' = q−1 pq
  Quaternion p_prime = quaternionMultiplication(
      quaternionMultiplication(quaternionInversion(q), tmp), q);

  // Perform passive rotation: when the coordinate system is rotated with
  // respect to the point. The two rotations are opposite from each other. p' =
  // qpq−1

  //  p_prime = quaternionMultiplication(quaternionMultiplication(q, tmp),
  //                                     quaternionInversion(q));

  // Step 3: Extract the rotated coordinates from p':
  Point rotate_d_p;
  rotate_d_p.x = p_prime.x;
  rotate_d_p.y = p_prime.y;
  rotate_d_p.z = p_prime.z;

  return rotate_d_p;
}

  
//rotate a vector3d with a quaternion
Eigen::Vector3d QuaternionRotation(Eigen::Quaterniond q, Eigen::Vector3d p)
{
    return   q *  p;    
}  


void QuaternionRotation() {

  Eigen::IOFormat HeavyFmt(Eigen::StreamPrecision, 2, ", ", ";\n", "[", "]",
                           "[", "]");

  // P  = [0, p1, p2, p3]  <-- point vector
  // alpha = angle to rotate
  //[x, y, z] = axis to rotate around (unit vector)
  // R = [cos(alpha/2), sin(alpha/2)*x, sin(alpha/2)*y, sin(alpha/2)*z] <-- rotation
  // R' = [w, -x, -y, -z]
  // P' = RPR'
  // P' = H(H(R, P), R')

  Eigen::Vector3d p(1, 0, 0);

  Quaternion P;
  P.w = 0;
  P.x = p(0);
  P.y = p(1);
  P.z = p(2);

  // rotation of 90 degrees about the y-axis
  double alpha = M_PI / 2;
  Quaternion R;
  Eigen::Vector3d r(0, 1, 0);
  r = r.normalized();

  R.w = cos(alpha / 2);
  R.x = sin(alpha / 2) * r(0);
  R.y = sin(alpha / 2) * r(1);
  R.z = sin(alpha / 2) * r(2);

  std::cout << R.w << "," << R.x << "," << R.y << "," << R.z << std::endl;

  Quaternion R_prime = quaternionInversion(R);
  Quaternion P_prime =
      quaternionMultiplication(quaternionMultiplication(R, P), R_prime);

  /*rotation of 90 degrees about the y-axis for the point (1, 0, 0). The result
  is (0, 0, -1). (Note that the first element of P' will always be 0 and can
  therefore be discarded.)
  */
  std::cout << P_prime.x << "," << P_prime.y << "," << P_prime.z << std::endl;
}


Quaternion rollPitchYawToQuaternion(double roll, double pitch,
                                    double yaw) // roll (x), pitch (Y), yaw (z)
{
  // Abbreviations for the various angular functions

  double cr = cos(roll * 0.5);
  double sr = sin(roll * 0.5);
  double cp = cos(pitch * 0.5);
  double sp = sin(pitch * 0.5);
  double cy = cos(yaw * 0.5);
  double sy = sin(yaw * 0.5);

  Quaternion q;
  q.w = cr * cp * cy + sr * sp * sy;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;

  return q;
}

void convertAxisAngleToQuaternion() {}

void ConvertQuaterniontoAxisAngle() {}

void QuaternionRepresentingRotationFromOneVectortoAnother() {
  Quaternion q;
  Eigen::Vector3d v1, v2;
  Eigen::Vector3d a = v1.cross(v2);
  q.x = a(0);
  q.y = a(1);
  q.z = a(2);

  q.w = sqrt((pow(v1.norm(), 2)) * (pow(v1.norm(), 2))) + v1.dot(v2);
}


int main() {

  //  double roll, pitch, yaw;

  //  roll = M_PI / 4;
  //  pitch = M_PI / 3;
  //  yaw = M_PI / 6;

  //  Eigen::Matrix3d rotationMatrix =
  //      eulerAnglesToRotationMatrix(roll, pitch, yaw);

  ////  Eigen::Quaterniond quaternionFromRotationMatrix(rotationMatrix);

  //  Eigen::Matrix3d R_b_c, R_s_b;

  //  R_b_c << 0, 0, -1, 0, 1, 0, 1, 0, 0;

  //  std::cout << "R_b_c\n" << R_b_c << std::endl;

  //  R_s_b << 0, -1, 0, 1, 0, 0, 0, 0, 1;

  //  std::cout << "R_s_b\n" << R_s_b << std::endl;

  //  Eigen::Matrix3d R_s_c = R_s_b * R_b_c;

  //  std::cout << "R_s_c\n" << R_s_c << std::endl;

  //  Eigen::Vector3d p_b;
  //  p_b << -1, 0, 0;

  //  Eigen::Vector3d p_s = R_s_b * p_b;

  //  std::cout << p_s << std::endl;


}
