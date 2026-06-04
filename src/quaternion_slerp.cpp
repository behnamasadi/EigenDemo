// Chapter 7: Spherical linear interpolation (SLERP) of unit quaternions.
//
// SLERP moves along the shortest great-circle arc between two orientations on
// the unit quaternion sphere, at constant angular velocity. It is the standard
// way to interpolate rotations smoothly (animation, trajectory generation),
// and avoids the artefacts of interpolating Euler angles or matrices directly.
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

int main() {
  // Two orientations: identity, and 90 degrees about the Z axis.
  Eigen::Quaterniond q0(Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()));
  Eigen::Quaterniond q1(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));

  // Equal steps in t produce equal steps in angle -> constant angular velocity.
  for (double t = 0.0; t <= 1.0001; t += 0.25) {
    Eigen::Quaterniond q = q0.slerp(t, q1);
    double angleDeg =
        2.0 * std::acos(std::min(1.0, std::abs(q.w()))) * 180.0 / M_PI;
    std::cout << "t=" << t << "  q=(" << q.w() << ", " << q.x() << ", " << q.y()
              << ", " << q.z() << ")   rotation about Z = " << angleDeg
              << " deg\n";
  }
  return 0;
}
