// A minimal first Eigen program: prints the detected Eigen version and does a
// small matrix/vector computation. If this builds and runs, your Eigen
// installation and CMake setup are working correctly.
#include <Eigen/Dense>
#include <iostream>

int main() {
  // Eigen exposes its version through preprocessor macros.
  std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "."
            << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\n\n";

  // A 2x2 matrix, filled with the comma initializer.
  Eigen::Matrix2d a;
  a << 1, 2, 3, 4;

  // A 2D column vector.
  Eigen::Vector2d b(5, 6);

  std::cout << "matrix a:\n" << a << "\n\n";
  std::cout << "vector b:\n" << b << "\n\n";
  std::cout << "a * b:\n" << a * b << std::endl;

  return 0;
}
