#include <Eigen/Dense>
#include <iostream>

void matrixCreation()
{
    Eigen::Matrix4d m; // 4x4 double

    Eigen::Matrix4cd objMatrix4cd; // 4x4 double complex


    //a is a 3x3 matrix, with a static float[9] array of uninitialized coefficients,
    Eigen::Matrix3f a;

    //b is a dynamic-size matrix whose size is currently 0x0, and whose array of coefficients hasn't yet been allocated at all.
    Eigen::MatrixXf b;

    //A is a 10x15 dynamic-size matrix, with allocated but currently uninitialized coefficients.
    Eigen::MatrixXf A(10, 15);
}

void arrayCreation()
{
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

void vectorCreation()
{
    // Vector3f is a fixed column vector of 3 floats:
    Eigen::Vector3f objVector3f;

    // RowVector2i is a fixed row vector of 3 integer:
    Eigen::RowVector2i objRowVector2i;

    // VectorXf is a column vector of size 10 floats:
    Eigen::VectorXf objv(10);

    //V is a dynamic-size vector of size 30, with allocated but currently uninitialized coefficients.
    Eigen::VectorXf V(30);
}

void buildMatrixFromVector()
{
    Eigen::Matrix2d mat;
    mat<<1,2,3,4;

    std::cout<<"matrix is:\n"  <<mat <<std::endl;

    Eigen::RowVector2d firstRow=mat.row(0);
    Eigen::Vector2d firstCol=mat.col(0);

    std::cout<<"First column of the matrix is:\n "<<firstCol <<std::endl;
    std::cout<<"First column dims are: "  <<firstCol.rows()<<","<<firstCol.cols() <<std::endl;


    std::cout<<"First row of the matrix is: \n" <<firstRow <<std::endl;
    std::cout<<"First row dims are: " <<firstRow.rows()<<"," <<firstRow.cols() <<std::endl;


    firstRow = Eigen::RowVector2d::Random();
    firstCol = Eigen::Vector2d::Random();


    mat.row(0) =firstRow;
    mat.col(0) = firstCol;

    std::cout<<"the new matrix is:\n"  <<mat <<std::endl;

}

void initialization()
{
    std::cout <<"///////////////////Initialization//////////////////"<< std::endl;

    Eigen::Matrix2d rndMatrix;
    rndMatrix.setRandom();

    Eigen::Matrix2d constantMatrix;
    constantMatrix.setRandom();
    constantMatrix.setConstant(4.3);

    Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(6,6);

    Eigen::MatrixXd zeros=Eigen::MatrixXd::Zero(3, 3);

    Eigen::ArrayXXf table(10, 4);
    table.col(0) = Eigen::ArrayXf::LinSpaced(10, 0, 90);




}

void elementAccess()
{
    std::cout <<"//////////////////Elements Access////////////////////"<< std::endl;

    Eigen::MatrixXf matrix(4, 4);
    matrix << 1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16;

    std::cout<<"matrix is:\n"<<matrix <<std::endl;

    std::cout<<"All Eigen matrices default to column-major storage order. That means, matrix(2) is  matix(2,0):" <<std::endl;

    std::cout<<"matrix(2): "<<matrix(2) <<std::endl;
    std::cout<<"matrix(2,0): "<<matrix(2,0) <<std::endl;


    std::cout <<"//////////////////Pointer to data ////////////////////"<< std::endl;

    for (int i = 0; i < matrix.size(); i++)
    {
          std::cout << *(matrix.data() + i) << "  ";
    }
    std::cout <<std::endl;


    std::cout <<"//////////////////Row major Matrix////////////////////"<< std::endl;
    Eigen::Matrix<double, 4,4,Eigen::RowMajor> matrixRowMajor(4, 4);
    matrixRowMajor << 1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16;


    for (int i = 0; i < matrixRowMajor.size(); i++)
    {
          std::cout << *(matrixRowMajor.data() + i) << "  ";
    }
    std::cout <<std::endl;


    std::cout <<"//////////////////Block Elements Access////////////////////"<< std::endl;

    std::cout << "Block elements in the middle" << std::endl;

    int starting_row,starting_column,number_rows_in_block,number_cols_in_block;

    starting_row=1;
    starting_column=1;
    number_rows_in_block=2;
    number_cols_in_block=2;

    std::cout << matrix.block(starting_row,starting_column,number_rows_in_block,number_cols_in_block) << std::endl;

    for (int i = 1; i <= 3; ++i)
    {
        std::cout << "Block of size " << i << "x" << i << std::endl;
        std::cout << matrix.block(0, 0, i, i) << std::endl;
    }
}

void matrixReshaping()
{
    //https://eigen.tuxfamily.org/dox/group__TutorialReshapeSlicing.html
    /*
    Eigen::MatrixXd m1(12,1);
    m1<<0,1,2,3,4,5,6,7,8,9,10,11;
    std::cout<<m1<<std::endl;

    //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> m2(m1);
    //Eigen::Map<Eigen::MatrixXd> m3(m2.data(),3,4);
    Eigen::Map<Eigen::MatrixXd> m2(m1.data(),4,3);
    std::cout<<m2.transpose()<<std::endl;
    //solution*/
    //https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
}

//https://eigen.tuxfamily.org/dox/group__TutorialReshapeSlicing.html
void matrixSlicing()
{

}
void matrixResizing()
{
    std::cout <<"//////////////////Matrix Resizing////////////////////"<< std::endl;
    int rows, cols;
    rows=3;
    cols=4;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dynamicMatrix;

    dynamicMatrix.resize(rows,cols);
    dynamicMatrix=Eigen::MatrixXd::Random(rows,cols);

    std::cout<<"Matrix size is: " << dynamicMatrix.size()<<std::endl;

    std::cout<<"Matrix is:\n" << dynamicMatrix<<std::endl;

    dynamicMatrix.resize(2,6);
    std::cout<<"New Matrix size is: " << dynamicMatrix.size()<<std::endl;
    std::cout<<"Matrix is:\n" << dynamicMatrix<<std::endl;

    std::cout <<"//////////////////Matrix conservativeResize////////////////////"<< std::endl;


    dynamicMatrix.conservativeResize(dynamicMatrix.rows(), dynamicMatrix.cols()+1);
    dynamicMatrix.col(dynamicMatrix.cols()-1) = Eigen::Vector2d(1, 4);

    std::cout<< dynamicMatrix<<std::endl;
}

void convertingMatrixtoArray()
{
    Eigen::Matrix<double, 4,4> mat1=Eigen::MatrixXd::Random(4,4);
    Eigen::Matrix<double, 4,4> mat2=Eigen::MatrixXd::Random(4,4);

    Eigen::Array<double, 4,4> array1=mat1.array();
    Eigen::Array<double, 4,4> array2=mat2.array();

    std::cout<<"Matrix multipication:\n" << mat1*mat2 <<std::endl;
    std::cout<<"Array multipication(coefficientsweise) :\n"<<array1*array2 <<std::endl;
    std::cout<<"Matrix coefficientsweise multipication :\n"<<mat1.cwiseProduct(mat2) <<std::endl;

}

void coefficientWiseOperations()
{
    std::cout <<"/////////////Matrix Coefficient Wise Operations///////////////////"<< std::endl;
    Eigen::Matrix<double, 2, 3> my_matrix;
    my_matrix << 1, 2, 3, 4, 5, 6;
    int i,j;
    std::cout <<"The matrix is: \n" << my_matrix << std::endl;

    std::cout <<"The matrix transpose is: \n" << my_matrix.transpose() << std::endl;

    std::cout<<"The minimum element is: " <<my_matrix.minCoeff(&i, &j)<<" and its indices are:" << i<<","<< j <<std::endl;


    std::cout<<"The maximum element is: " <<my_matrix.maxCoeff(&i, &j)<<" and its indices are:" << i<<","<< j <<std::endl;

    std::cout<<"The multipication of all elements: " <<my_matrix.prod()<<std::endl;
    std::cout<<"The sum of all elements: " <<my_matrix.sum()<<std::endl;
    std::cout<<"The mean of all element: " <<my_matrix.mean()<<std::endl;
    std::cout<<"The trace of the matrix is: " <<my_matrix.trace()<<std::endl;
    std::cout<<"The means of columns: " <<my_matrix.colwise().mean()<<std::endl;
    std::cout<<"The max of each columns: "  <<my_matrix.rowwise().maxCoeff()<<std::endl;
    std::cout<<"Norm 2 of the matrix is: " <<my_matrix.lpNorm<2>()<<std::endl;
    std::cout<<"Norm infinty of the matrix is: " <<my_matrix.lpNorm<Eigen::Infinity>()<<std::endl;
    std::cout<<"If all elemnts are positive: "  << (my_matrix.array()>0).all()<<std::endl;
    std::cout<<"If any element is greater than 2: "<<(my_matrix.array()>2).any()<<std::endl;
    std::cout<<"Counting the number of elements greater than 1"<<(my_matrix.array()>1).count()<<std::endl;
    std::cout <<"subtracting 2 from all elements:\n" << my_matrix.array() - 2 << std::endl;
    std::cout <<"abs of the matrix: \n" << my_matrix.array().abs() << std::endl;
    std::cout << "square of the matrix: \n" <<my_matrix.array().square() << std::endl;

    std::cout <<"exp of the matrix: \n" <<my_matrix.array().exp() << std::endl;
    std::cout << "log of the matrix: \n" <<my_matrix.array().log() << std::endl;
    std::cout << "square root of the matrix: \n" <<my_matrix.array().sqrt() << std::endl;





}

void maskingArray()
{
    std::cout <<"//////////////////Maskin Matrix/Array ////////////////////"<< std::endl;

    //Eigen::MatrixXf P, Q, R; // 3x3 float matrix.
    // (R.array() < s).select(P,Q ); // (R < s ? P : Q)
    // R = (Q.array()==0).select(P,R); // R(Q==0) = P(Q==0)
    int cols, rows;
    cols=2; rows=3;
    Eigen::MatrixXf R=Eigen::MatrixXf::Random(rows, cols);

    Eigen::MatrixXf Q=Eigen::MatrixXf::Zero(rows, cols);
    Eigen::MatrixXf P=Eigen::MatrixXf::Constant(rows, cols,1.0);

    double s=0.5;
    Eigen::MatrixXf masked=(R.array() < s).select(P,Q ); // (R < s ? P : Q)

    std::cout<<"R\n" <<R <<std::endl;
    std::cout<<"masked\n" <<masked <<std::endl;
    std::cout<<"P\n"<< P <<std::endl;
    std::cout<<"Q\n" << Q<<std::endl;
}

void AdditionSubtractionOfMatrices()
{

}



void transpositionConjugation()
{

}


void scalarMultiplicationDivision()
{

}


void multiplicationDotCrossProduct()
{

}

int main()
{
    //buildMatrixFromVector();
    //vectorCreation();
    //elementAccess();
    //matrixResizing();
    //coefficientWiseOperations();
    //convertingMatrixtoArray();
    //maskingArray();

}
