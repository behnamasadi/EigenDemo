#include <Eigen/Dense>
#include <iostream>

void matrixCreation() {
  Eigen::Matrix4d m; // 4x4 double

  Eigen::Matrix4cd objMatrix4cd; // 4x4 double complex

  // a is a 3x3 matrix, with a static float[9] array of uninitialized
  // coefficients,
  Eigen::Matrix3f a;

  // b is a dynamic-size matrix whose size is currently 0x0, and whose array of
  // coefficients hasn't yet been allocated at all.
  Eigen::MatrixXf b;

  // A is a 10x15 dynamic-size matrix, with allocated but currently
  // uninitialized coefficients.
  Eigen::MatrixXf A(10, 15);
}

void arrayCreation() {
  // ArrayXf
  Eigen::Array<float, Eigen::Dynamic, 1> a1;
  // Array3f
  Eigen::Array<float, 3, 1> a2;
  // ArrayXXd
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> a3;
  // Array33d
  Eigen::Array<double, 3, 3> a4;
  Eigen::Matrix3d matrix_from_array = a4.matrix();
}

void vectorCreation() {
  // Vector3f is a fixed column vector of 3 floats:
  Eigen::Vector3f objVector3f;

  // RowVector2i is a fixed row vector of 3 integer:
  Eigen::RowVector2i objRowVector2i;

  // VectorXf is a column vector of size 10 floats:
  Eigen::VectorXf objv(10);

  // V is a dynamic-size vector of size 30, with allocated but currently
  // uninitialized coefficients.
  Eigen::VectorXf V(30);
}

void buildMatrixFromVector() {
  Eigen::Matrix2d mat;
  mat << 1, 2, 3, 4;

  std::cout << "matrix is:\n" << mat << std::endl;

  Eigen::RowVector2d firstRow = mat.row(0);
  Eigen::Vector2d firstCol = mat.col(0);

  std::cout << "First column of the matrix is:\n " << firstCol << std::endl;
  std::cout << "First column dims are: " << firstCol.rows() << ","
            << firstCol.cols() << std::endl;

  std::cout << "First row of the matrix is: \n" << firstRow << std::endl;
  std::cout << "First row dims are: " << firstRow.rows() << ","
            << firstRow.cols() << std::endl;

  firstRow = Eigen::RowVector2d::Random();
  firstCol = Eigen::Vector2d::Random();

  mat.row(0) = firstRow;
  mat.col(0) = firstCol;

  std::cout << "the new matrix is:\n" << mat << std::endl;
}

void initialization() {
  std::cout << "///////////////////Initialization//////////////////"
            << std::endl;

  Eigen::Matrix2d rndMatrix;
  rndMatrix.setRandom();

  Eigen::Matrix2d constantMatrix;
  constantMatrix.setRandom();
  constantMatrix.setConstant(4.3);

  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(6, 6);

  Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(3, 3);

  Eigen::ArrayXXf table(10, 4);
  table.col(0) = Eigen::ArrayXf::LinSpaced(10, 0, 90);
}

void elementAccess() {
  std::cout << "//////////////////Elements Access////////////////////"
            << std::endl;

  Eigen::MatrixXf matrix(4, 4);
  matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;

  std::cout << "matrix is:\n" << matrix << std::endl;

  std::cout << "All Eigen matrices default to column-major storage order. That "
               "means matrix(2) is the same as matrix(2,0):"
            << std::endl;

  std::cout << "matrix(2): " << matrix(2) << std::endl;
  std::cout << "matrix(2,0): " << matrix(2, 0) << std::endl;

  std::cout << "//////////////////Pointer to data ////////////////////"
            << std::endl;

  for (int i = 0; i < matrix.size(); i++) {
    std::cout << *(matrix.data() + i) << "  ";
  }
  std::cout << std::endl;

  std::cout << "//////////////////Row major Matrix////////////////////"
            << std::endl;
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> matrixRowMajor(4, 4);
  matrixRowMajor << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;

  for (int i = 0; i < matrixRowMajor.size(); i++) {
    std::cout << *(matrixRowMajor.data() + i) << "  ";
  }
  std::cout << std::endl;

  std::cout << "//////////////////Block Elements Access////////////////////"
            << std::endl;

  std::cout << "Block elements in the middle" << std::endl;

  int starting_row, starting_column, number_rows_in_block, number_cols_in_block;

  starting_row = 1;
  starting_column = 1;
  number_rows_in_block = 2;
  number_cols_in_block = 2;

  std::cout << matrix.block(starting_row, starting_column, number_rows_in_block,
                            number_cols_in_block)
            << std::endl;

  for (int i = 1; i <= 3; ++i) {
    std::cout << "Block of size " << i << "x" << i << std::endl;
    std::cout << matrix.block(0, 0, i, i) << std::endl;
  }
}

// Reshaping (Eigen 3.4+): .reshaped() returns a view of the same coefficients
// with different dimensions. The view is column-major by default.
// https://eigen.tuxfamily.org/dox/group__TutorialReshapeSlicing.html
void matrixReshaping() {
  std::cout << "//////////////////Matrix Reshaping////////////////////"
            << std::endl;
  Eigen::MatrixXd m(2, 3);
  m << 1, 2, 3,
       4, 5, 6;
  std::cout << "original 2x3 matrix:\n" << m << std::endl;
  std::cout << "reshaped to 3x2 (column-major order):\n"
            << m.reshaped(3, 2) << std::endl;
  std::cout << "flattened to a single row:\n"
            << m.reshaped().transpose() << std::endl;
}

// Slicing (Eigen 3.4+): index a matrix with seq()/seqN()/all/last to extract
// arbitrary sub-blocks, strided ranges, or whole rows/columns.
// https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html
void matrixSlicing() {
  std::cout << "//////////////////Matrix Slicing////////////////////"
            << std::endl;
  Eigen::MatrixXd m(4, 4);
  m << 1, 2, 3, 4,
       5, 6, 7, 8,
       9, 10, 11, 12,
       13, 14, 15, 16;

  std::cout << "rows 1..2, all columns:\n"
            << m(Eigen::seq(1, 2), Eigen::all) << std::endl;
  std::cout << "every other row, all columns:\n"
            << m(Eigen::seq(0, Eigen::last, 2), Eigen::all) << std::endl;
  std::cout << "the last column:\n" << m(Eigen::all, Eigen::last) << std::endl;
}
void matrixResizing() {
  std::cout << "//////////////////Matrix Resizing////////////////////"
            << std::endl;
  int rows, cols;
  rows = 3;
  cols = 4;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dynamicMatrix;

  dynamicMatrix.resize(rows, cols);
  dynamicMatrix = Eigen::MatrixXd::Random(rows, cols);

  std::cout << "Matrix size is: " << dynamicMatrix.size() << std::endl;

  std::cout << "Matrix is:\n" << dynamicMatrix << std::endl;

  dynamicMatrix.resize(2, 6);
  std::cout << "New Matrix size is: " << dynamicMatrix.size() << std::endl;
  std::cout << "Matrix is:\n" << dynamicMatrix << std::endl;

  std::cout << "//////////////////Matrix conservativeResize////////////////////"
            << std::endl;

  dynamicMatrix.conservativeResize(dynamicMatrix.rows(),
                                   dynamicMatrix.cols() + 1);
  dynamicMatrix.col(dynamicMatrix.cols() - 1) = Eigen::Vector2d(1, 4);

  std::cout << dynamicMatrix << std::endl;
}

void convertingMatrixtoArray() {
  Eigen::Matrix<double, 4, 4> mat1 = Eigen::MatrixXd::Random(4, 4);
  Eigen::Matrix<double, 4, 4> mat2 = Eigen::MatrixXd::Random(4, 4);

  Eigen::Array<double, 4, 4> array1 = mat1.array();
  Eigen::Array<double, 4, 4> array2 = mat2.array();

  std::cout << "Matrix multiplication:\n" << mat1 * mat2 << std::endl;
  std::cout << "Array multiplication (coefficient-wise):\n"
            << array1 * array2 << std::endl;
  std::cout << "Matrix coefficient-wise multiplication:\n"
            << mat1.cwiseProduct(mat2) << std::endl;
}

void convertingArrayToMatrix() {
  std::cout << "//////////////////Array <-> Matrix////////////////////"
            << std::endl;
  Eigen::Array<double, 4, 4> array1 = Eigen::Array<double, 4, 4>::Random();
  Eigen::Matrix<double, 4, 4> mat1 = array1.matrix();

  std::cout << "array1:\n" << array1 << std::endl;
  std::cout << "array1 viewed as a matrix:\n" << mat1 << std::endl;
}

void castingMatrices() {
  std::cout << "//////////////////Casting Matrices////////////////////"
            << std::endl;

  Eigen::Matrix<float, 2, 3> matrix_23;
  matrix_23 << 1, 2, 3, 4, 5, 6;
  Eigen::Vector3d v_3d(1, 2, 3);

  // This would NOT compile: a float matrix cannot multiply a double vector.
  // Eigen::Matrix<double, 2, 1> wrong = matrix_23 * v_3d;

  // Cast the float matrix to double so both operands share the scalar type:
  Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;

  std::cout << "matrix_23.cast<double>() * v_3d =\n" << result << std::endl;
}

int main() {
  matrixCreation();
  arrayCreation();
  vectorCreation();
  buildMatrixFromVector();
  initialization();
  elementAccess();
  convertingArrayToMatrix();
  convertingMatrixtoArray();
  castingMatrices();
  matrixResizing();
  matrixReshaping();
  matrixSlicing();
}
