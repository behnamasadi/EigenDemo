#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

//CRS
void compressedSparseRow()
{
/*
    0 1 2 3 4 5 6 7 8
   ┌                 ┐
 0 |0 0 0 0 0 0 0 3 0|
 1 |0 0 8 0 0 1 0 0 0|
 2 |0 0 0 0 0 0 0 0 0|
 3 |4 0 0 0 0 0 0 0 0|
 4 |0 0 0 0 0 0 0 0 0|
 5 |0 0 2 0 0 0 0 0 0|
 6 |0 0 0 6 0 0 0 0 0|
 7 |0 9 0 0 5 0 0 0 0|
   └                 ┘
*/


    int rows=8;
    int cols=9;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> dense_mat;
    dense_mat.resize(rows,cols);


    dense_mat<<0, 0, 0, 0, 0, 0, 0, 3, 0,
                0, 0, 8, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 2, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 6, 0, 0, 0, 0, 0,
                0, 9, 0, 0, 5, 0, 0, 0, 0;

    //specify a tolerance:
    //sparse_mat = dense_mat.sparseView(epsilon,reference);
    Eigen::SparseMatrix<double,Eigen::RowMajor> sparse_mat = dense_mat.sparseView();

    std::cout<<"sparse matrix number of cols:"<<sparse_mat.cols()   <<std::endl;
    std::cout<<"sparse matrix numner of rows:"<<sparse_mat.rows()   <<std::endl;

    std::cout<<"sparse matrix values:"<<std::endl;
    auto valuePtr=sparse_mat.valuePtr();
    for(int i=0;i <sparse_mat.nonZeros();i++)
    {
        std::cout<<*(valuePtr++)  <<", ";
    }
    std::cout<< "\n";


    std::cout<<"COL_INDEX array size:"<<sparse_mat.innerSize()  <<std::endl;
    std::cout<<"COL_INDEX array :" <<std::endl;
    auto colIndexPtr=sparse_mat.innerIndexPtr();

    for(int i=0;i <sparse_mat.innerSize();i++)
    {
        std::cout<<*(colIndexPtr++)  <<", ";
    }
    std::cout<< "\n";

    std::cout<<"ROW_INDEX array size:"<<sparse_mat.outerSize()   <<std::endl;
    std::cout<<"ROW_INDEX array:" <<std::endl;
    auto rowIndexPtr=sparse_mat.outerIndexPtr();
    for(int i=0;i <sparse_mat.outerSize();i++)
    {
        std::cout<<*(rowIndexPtr++)  <<", ";
    }
    std::cout<< "\n ";

    std::cout<<"*********sparse matrix*********\n"<<sparse_mat  <<std::endl;

    std::cout<<"*********dense matrix*********\n"<<dense_mat  <<std::endl;






}

int main1()
{
    compressedSparseRow();
}



typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;//structure to hold a non zero as a triplet (i,j,value).

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n){}
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename){}
//https://eigen.tuxfamily.org/dox/classEigen_1_1Triplet.html
int main(int argc, char** argv)
{
  if(argc!=2) {
    std::cerr << "Error: expected one and only one argument.\n";
    return -1;
  }

  int n = 300;  // size of the image
  int m = n*n;  // number of unknows (=number of pixels)

  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  buildProblem(coefficients, b, n);

  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
  Eigen::VectorXd x = chol.solve(b);         // use the factorization to solve for the given right hand side

  // Export the result to a file:
  saveAsBitmap(x, n, argv[1]);

  return 0;
}
