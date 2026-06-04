// Chapter 3: Checking matrix similarity.
//
// Floating-point matrices should never be compared with ==, because rounding
// makes exact equality unreliable. Use isApprox(), which tests equality up to a
// relative precision, or compare the norm of the difference against a
// tolerance.
#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::Vector3d a(1.0, 2.0, 3.0);
  Eigen::Vector3d b(1.0, 2.0, 3.0 + 1e-12);

  std::cout << std::boolalpha;
  std::cout << "a == b (exact):        " << (a == b) << "\n";
  std::cout << "a.isApprox(b):         " << a.isApprox(b) << "\n";
  std::cout << "a.isApprox(b, 1e-6):   " << a.isApprox(b, 1e-6) << "\n";

  const double precision = 1e-6;
  std::cout << "(a - b).norm() < tol:  " << ((a - b).norm() < precision)
            << "\n";

  return 0;
}
