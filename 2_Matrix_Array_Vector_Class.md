- [Matrix Class](#matrix-class)
- [Vector Class](#vector-class)
- [Array Class](#array-class)
  * [Converting Array to Matrix](#converting-array-to-matrix)
  * [Converting Matrix to Array](#converting-matrix-to-array)
- [Initialization](#initialization)
- [Accessing Elements (Coefficient)](#accessing-elements--coefficient-)
  * [Accessing via parenthesis](#accessing-via-parenthesis)
  * [Accessing via pointer to data](#accessing-via-pointer-to-data)
  * [Row Major Access](#row-major-access)
  * [Accessing a block of data](#accessing-a-block-of-data)
- [Reshaping, Resizing, Slicing](#reshaping--resizing--slicing)
- [Tensor Module](#tensor-module)

# Matrix Class
The Matrix class takes six template parameters, the first mandatory three first parameters are:

```
Eigen::Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
```

For instance: 
```
Eigen::Matrix<double, 2, 1> matrix;

```
If dimensions are known at compile time, you can use `Eigen::Dynamic` as the template parameter. for instance
```
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>  matrix;
```

Eigen offers a lot of convenience `typedefs` to cover the usual cases, the definition of these matrices (also vectors and arrays) in Eigen comes in the following form:
```
Eigen::MatrixSizeType
Eigen::VectorSizeType
Eigen::ArraySizeType
```
Type can be:
- i for integer,
- f for float,
- d for double,
- c for complex,
- cf for complex float,
- cd for complex double.

Size can be 2,3,4 for fixed size square matrices or `X` for dynamic size

For example, `Matrix4f` is a `4x4` matrix of floats. Here is how it is defined by Eigen:
```
typedef Matrix<float, 4, 4> Matrix4f;
```
or 
```
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
``` 
 
 
Here are more examples:

```
Eigen::Matrix4d m; // 4x4 double

Eigen::Matrix4cd objMatrix4cd; // 4x4 double complex


//a is a 3x3 matrix, with a static float[9] array of uninitialized coefficients,
Eigen::Matrix3f a;

//b is a dynamic-size matrix whose size is currently 0x0, and whose array of coefficients hasn't yet been allocated at all.
Eigen::MatrixXf b;

//A is a 10x15 dynamic-size matrix, with allocated but currently uninitialized coefficients.
Eigen::MatrixXf A(10, 15);
```



# Vector Class
Vectors are single row/ single column matrices. For instance:
```
// Vector3f is a fixed column vector of 3 floats:
Eigen::Vector3f objVector3f;

// RowVector2i is a fixed row vector of 3 integer:
Eigen::RowVector2i objRowVector2i;

// VectorXf is a column vector of size 10 floats:
Eigen::VectorXf objv(10);

//V is a dynamic-size vector of size 30, with allocated but currently uninitialized coefficients.
Eigen::VectorXf V(30);
```
You get/ set each row or column of marix with/ from a vector:

```
Eigen::Matrix2d mat;
mat<<1,2,3,4;
Eigen::RowVector2d firstRow=mat.row(0);
Eigen::Vector2d firstCol=mat.col(0);


firstRow = Eigen::RowVector2d::Random();
firstCol = Eigen::Vector2d::Random();


mat.row(0) =firstRow;
mat.col(0) = firstCol;
```


# Array Class
Eigen Matrix class is intended for linear algebra. The Array class is the general-purpose arrays, 
which has coefficient-wise operations, such as adding a constant to every coefficient, multiplying two arrays coefficient-wise, etc.

```
// ArrayXf
Eigen::Array<float, Eigen::Dynamic, 1> a1;
// Array3f
Eigen::Array<float, 3, 1> a2;
// ArrayXXd
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> a3;
// Array33d
Eigen::Array<double, 3, 3> a4;
Eigen::Matrix3d matrix_from_array = a4.matrix();
```

## Converting Array to Matrix
```
Eigen::Array<double, 4,4> array1=Eigen::ArrayXd::Random(4,4);
Eigen::Array<double, 4,4> mat1=array1.matrix();
```

## Converting Matrix to Array
```
Eigen::Matrix<double, 4,4> mat1=Eigen::MatrixXd::Random(4,4);
Eigen::Array<double, 4,4> array1=mat1.array();
```



# Initialization
You can initialize your matrix with comma
```
Eigen::Matrix<double, 2, 3> matrix;
matrix << 1, 2, 3, 4, 5, 6;
```

There are various out of the box APIs for special matrics, for intance

```
Eigen::Matrix2d rndMatrix;
rndMatrix.setRandom();

Eigen::Matrix2d constantMatrix;
constantMatrix.setConstant(4.3);

Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(6,6);

Eigen::MatrixXd zeros=Eigen::MatrixXd::Zero(3, 3);

Eigen::ArrayXXf table(10, 4);
table.col(0) = Eigen::ArrayXf::LinSpaced(10, 0, 90);
```


# Accessing Elements (Coefficient)
## Accessing via parenthesis
Eigen has overloaded parenthesis operators, that means you can access the elements with row and column index: `matrix(row,col)`.

All Eigen matrices default to column-major storage order. That means, `matrix(2)` is the the third element of first column matix(2,0):


```
Eigen::MatrixXf matrix(4, 4);
    matrix << 1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16;
matrix(2): "<<matrix(2) 
matrix(2,0): "<<matrix(2,0) 
```
## Accessing via pointer to data

If you need to access the underlying data directly, you can use `matrix.data()` which returns a pointer to teh first element:

```
   for (int i = 0; i < matrix.size(); i++)
    {
          std::cout << *(matrix.data() + i) << "  ";
    }
```    
## Row Major Access

By default the, Eigen matrices are column major, to change it just pass the template parameter:
```
Eigen::Matrix<double, 4,4,Eigen::RowMajor> matrixRowMajor(4, 4);
```

## Accessing a block of data

You can have access to a block in a matrix.
```
int starting_row,starting_column,number_rows_in_block,number_cols_in_block;

starting_row=1;
starting_column=1;
number_rows_in_block=2;
number_cols_in_block=2;
matrix.block(starting_row,starting_column,number_rows_in_block,number_cols_in_block);
```


# Reshaping, Resizing, Slicing
The current size of a matrix can be retrieved by `rows()`, `cols()` and `size()` method. Resizing a dynamic-size matrix is done by the resize() method, which is a no-operation if the actual matrix size doesn't change, i.e. chaning from `3x4` to `6x2`

```
int rows, cols;
rows=3;
cols=4;
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dynamicMatrix;

dynamicMatrix.resize(rows,cols);
dynamicMatrix=Eigen::MatrixXd::Random(rows,cols);
dynamicMatrix.resize(2,6);
```

If your new size is different than matrix size, and you want a conservative your data, you should use `conservativeResize()` method
```
dynamicMatrix.conservativeResize(dynamicMatrix.rows(), dynamicMatrix.cols()+1);
dynamicMatrix.col(dynamicMatrix.cols()-1) = Eigen::Vector2d(1, 4);
```


# Tensor Module








