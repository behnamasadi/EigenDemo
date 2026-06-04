// Chapter 10: PCA via eigendecomposition -- fitting a plane to a point cloud.
//
// The best-fit plane through a set of points passes through their centroid, and
// its normal is the direction of least variance -- the eigenvector of the
// smallest eigenvalue of the (symmetric, positive-semidefinite) covariance
// matrix. SelfAdjointEigenSolver returns eigenvalues in ascending order, so the
// first eigenvector is the normal. This is how surface normals are estimated in
// point-cloud processing.
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>

int main() {
  // Points scattered around the plane z = 0 (true normal ~ (0,0,1)) + noise.
  Eigen::Matrix<double, 3, 8> pts;
  pts << -1.0, 1.0, -1.0, 1.0, 2.0, -2.0, 0.0, 0.5, -1.0, -1.0, 1.0, 1.0, 0.0,
      1.0, 2.0, -1.0, 0.02, -0.01, 0.0, 0.03, -0.02, 0.01, 0.0, -0.03;

  Eigen::Vector3d centroid = pts.rowwise().mean();
  Eigen::Matrix<double, 3, 8> centered = pts.colwise() - centroid;
  Eigen::Matrix3d cov = centered * centered.transpose(); // 3x3 symmetric PSD

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
  Eigen::Vector3d normal = es.eigenvectors().col(0); // smallest eigenvalue

  std::cout << "eigenvalues (ascending): " << es.eigenvalues().transpose()
            << "\n";
  std::cout << "plane normal (least-variance direction): " << normal.transpose()
            << "\n";
  std::cout << "centroid (a point on the plane):         "
            << centroid.transpose() << "\n";
  return 0;
}
