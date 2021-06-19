#include <iostream>
#include <vector>
#include <iomanip>
#include <Eigen/Dense>



double exp(double x) // the functor we want to apply
{
    std::setprecision(5);
        return std::trunc(x);
}


void MatrixPowersPolynomials()
{

    //Rectangular Diagonal
    // rectangular diagonal  matrix is an n × d matrix in which each entry (i, j) has a non-zero value if and  only if i = j.
    // diagonal matrix is a matrix in which the entries outside the main diagonal are all zero;


    //A block diagonal matrix contains square blocks B 1…B r of (possibly) nonzero entries along the diagonal. All other entries are zero. Although each
    //block is square, they need not be of the same size.

    //Upper and Lower Triangular Matrix) A square matrix is an upper triangular matrix if all entries (i, j) below its main diagonal (i.e.,
    //satisfying i > j) are zeros.

    // The product of uppertriangular matrices is upper triangular.
    //c(i,j)=0 if i>j c(i,j)=sum(a(i,k)*b(k,j))
    //i>j
    //if i>k a(i,k)=0
    //if i<k -> j<k -> b(k,j)=0

    //Inverse of Triangular Matrix Is Triangula

    //Strictly Triangular Matrix) A matrix is said to be strictlytriangular if it is triangularandall its diagonal elements are zeros.

    //The zeroth power of a matrix is defined to be the identity matrix
    //When a matrix satisfies A^k = 0 for some integer k, it is referred to as nilpotent.
    //all strictly triangular matrices (triangular with zero main diagonal)of size d × d satisfy A^d = 0.
    //product of two upper triangular is a triangular matrix

    //polynomial function f(A) of a square matrix in  much the same way as one computes polynomials of scalars.
    //f(x) = 3x^2 + 5x + 2 -> f(A) = 3A^2 + 5A + 2
    //Two    polynomials f(A) and g(A) of the same matrix A will always commute f(A)g(A)=g(A)f(A)

    //Commutativity of Multiplication with Inverse if AB=I then BA=I
    //When the inverse of a matrix exists, it is always unique
    //Inverse of Triangular Matrix Is Triangular
    //Inv(A^n)=Inv(A)^n

    //An orthogonal matrix is a square matrix whose inverse is its transpose: A*A^T=A^T*A=I
    //all column columns/ rows are perpendicular

    //The multiplication of an n × d matrix A with a d-dimensional column
    //vector to create an n-dimensional column vector is often interpreted as
    //a linear transformation from d-dimensional space to n-dimensional space
/*
    a11   a12         a11      a12
    a21   a22  x1 =x1 a21 + x2 a22
    a31   a32  x2     a31      a32

Therefore, the n × d matrix A is occasionally represented in terms of its ordered set of ndimensional columns
A nxd= [a1 a2 ... ad]

 low-rank update
 Linear regression least-squares classification, support-vector machines, and logistic regression
 Matrix factorization is an alternative term for matrix decomposition
 recommender systems

Regression
the only difference from classification is that the array contains numerical values (rather than categorical ones)
The dependent
variable is also referred to as a response variable, target variable, or regressand in the case of regression
The independent variables are also referred to as regressors.more than two classes like {Red,
Green, Blue} cannot be ordered, and are therefore different from regression


convex objective functions like linear regression

https://medium.com/@wisnutandaseru/proving-eulers-identity-using-taylor-series-2771089cd780
Householder Reflections
Directional derivative
chain rule for multivariale derivative
multivariant fourier transform



inverse of I+A?
https://math.stackexchange.com/questions/298616/what-is-inverse-of-ia/298623
https://en.wikipedia.org/wiki/Woodbury_matrix_identity


https://en.m.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra
*/

}


int main(int argc, char *argv[])
{
    return 0;
}

