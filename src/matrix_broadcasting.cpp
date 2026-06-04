// Chapter 3: Broadcasting.
// Broadcasting replicates a vector along the rows or columns of a matrix so it
// can be combined with the whole matrix at once, similar to NumPy broadcasting.
// https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
#include <Eigen/Dense>
#include <iostream>

void broadcastingExample() {
  Eigen::MatrixXf mat(2, 4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;

  // Add a column vector to every column.
  Eigen::VectorXf v(2);
  v << 0, 1;
  std::cout << "mat =\n" << mat << "\n";
  std::cout << "add v to each column (mat.colwise() + v):\n"
            << (mat.colwise() + v) << "\n";

  // Add a row vector to every row.
  Eigen::RowVectorXf w(4);
  w << 0, 1, 2, 3;
  std::cout << "add w to each row (mat.rowwise() + w):\n"
            << (mat.rowwise() + w) << "\n";

  // A classic use of broadcasting: find the column closest to a target vector.
  Eigen::VectorXf target(2);
  target << 5, 4;
  Eigen::Index index;
  float minSquaredDist =
      (mat.colwise() - target).colwise().squaredNorm().minCoeff(&index);
  std::cout << "nearest column to target is column " << index
            << " (squared distance " << minSquaredDist << ")\n";
}

int main() { broadcastingExample(); }
