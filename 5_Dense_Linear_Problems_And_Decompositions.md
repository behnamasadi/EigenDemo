- [Chapter 5 Dense Linear Problems And Decompositions](#chapter-5-dense-linear-problems-and-decompositions)
- [1. Vector Space](#1-vector-space)
  * [1.1. Examples of Vector Spaces](#11-examples-of-vector-spaces)
  * [1.2. Vector Products](#12-vector-products)
    + [1.2.1 Dot Product](#121-dot-product)
    + [1.2.2 The Hadamard product (Schur product)](#122-the-hadamard-product--schur-product-)
    + [1.2.3 Kronecker product](#123-kronecker-product)
- [2. Linear Equation](#2-linear-equation)
  * [2.1. Intuition behind Matrix Multiplication:](#21-intuition-behind-matrix-multiplication-)
  * [2.2. Solution set](#22-solution-set)
  * [2.3. Underdetermined System](#23-underdetermined-system)
  * [2.4. Overdetermined System](#24-overdetermined-system)
  * [2.5. Determined](#25-determined)
  * [2.6. Homogeneous vs Non-homogeneous](#26-homogeneous-vs-non-homogeneous)
- [3. Solving Linear Equation](#3-solving-linear-equation)
  * [3.1. Using the SVD](#31-using-the-svd)
  * [3.2. Complete Orthogonal Decomposition](#32-complete-orthogonal-decomposition)
  * [3.3. Using the QR](#33-using-the-qr)
  * [3.4.  Using Cholesky Decomposition](#34--using-cholesky-decomposition)
  * [3.5. Gaussian Elimination (row reduction)](#35-gaussian-elimination--row-reduction-)
    + [3.5.1. Forward Elimination](#351-forward-elimination)
    + [3.5.2. Forward and Back Substitution](#352-forward-and-back-substitution)
    + [3.5.3. Partial Pivoting and Full Pivoting](#353-partial-pivoting-and-full-pivoting)
    + [3.5.4. Numerical stability in Gaussian Elimination](#354-numerical-stability-in-gaussian-elimination)
    + [3.5.5. Example of The Gaussian Elimination Algorithm](#355-example-of-the-gaussian-elimination-algorithm)
    + [3.5.6.  Row Echelon Form](#356--row-echelon-form)
    + [3.5.7.  Reduced Row Echelon Form](#357--reduced-row-echelon-form)
    + [3.5.8.  Example of Row Echelon](#358--example-of-row-echelon)
    + [3.5.9. Echelon Pivot Column](#359-echelon-pivot-column)
- [4. Matrices Decompositions](#4-matrices-decompositions)
  * [4.1. QR Decomposition](#41-qr-decomposition)
    + [4.1.1. Square Matrix QR Decomposition](#411-square-matrix-qr-decomposition)
    + [4.1.2.  Rectangular Matrix QR Decomposition](#412--rectangular-matrix-qr-decomposition)
  * [4.2. Computing the QR Decomposition](#42-computing-the-qr-decomposition)
  * [4.3. Gram Schmidt Orthogonalization](#43-gram-schmidt-orthogonalization)
  * [4.4. Householder Transformations](#44-householder-transformations)
  * [4.5. QL, RQ and LQ Decompositions](#45-ql--rq-and-lq-decompositions)
  * [4.6. Cholesky Decomposition LL*](#46-cholesky-decomposition-ll-)
  * [4.7. Hermitian Matrix](#47-hermitian-matrix)
  * [4.8. Positive (Semidefinite) Definite Matrix:](#48-positive--semidefinite--definite-matrix-)
  * [4.9. LDL Decomposition](#49-ldl-decomposition)
  * [4.10. Lower Upper (LU) Decomposition](#410-lower-upper--lu--decomposition)
  * [4.11. Lower Diagonal Upper (LDU) decomposition](#411-lower-diagonal-upper--ldu--decomposition)
  * [4.12. Eigen Value and Eigen Vector](#412-eigen-value-and-eigen-vector)
  * [4.13. Calculation of Eigen Value and Eigen Vector](#413-calculation-of-eigen-value-and-eigen-vector)
    + [4.13.1  Example of Calculating Eigen Value and Eigen Vector](#4131--example-of-calculating-eigen-value-and-eigen-vector)
  * [4.14. Eigendecomposition of Matrix](#414-eigendecomposition-of-matrix)
  * [4.15. Singular Value Decomposition](#415-singular-value-decomposition)
    + [4.15.1 Applications of the SVD](#4151-applications-of-the-svd)
      - [4.15.1.1 Pseudoinverse](#41511-pseudoinverse)
      - [4.15.1.2 Solving homogeneous linear equations](#41512-solving-homogeneous-linear-equations)
      - [4.15.1.3  Range, null space and rank](#41513--range--null-space-and-rank)
      - [4.15.1.4 Nearest orthogonal matrix](#41514-nearest-orthogonal-matrix)
- [5. Linear Map](#5-linear-map)
- [6. Span](#6-span)
- [7. Subspace](#7-subspace)
- [7.1. Row Spaces and Column Spaces](#71-row-spaces-and-column-spaces)
- [8. Range of a Matrix](#8-range-of-a-matrix)
  * [8.1. Example of Row Spaces](#81-example-of-row-spaces)
- [9. Basis](#9-basis)
  * [9.1. Example of Computing Basis for Column Space](#91-example-of-computing-basis-for-column-space)
  * [9.2. Example of Computing Basis for Row Space](#92-example-of-computing-basis-for-row-space)
  * [9.3. Changes of basis vectors](#93-changes-of-basis-vectors)
  * [9.4. Covariance and Contravariance of Vectors](#94-covariance-and-contravariance-of-vectors)
  * [9.5. Creating a Basis Set](#95-creating-a-basis-set)
  * [9.6. Change of Basis](#96-change-of-basis)
  * [9.7. Vector Fields](#97-vector-fields)
  * [9.8. Coordinate System](#98-coordinate-system)
    + [9.8.1. Cartesian, Polar, Curvilinear coordinates ,Cylindrical and Spherical Coordinates](#981-cartesian--polar--curvilinear-coordinates--cylindrical-and-spherical-coordinates)
  * [9.9. Coordinate transformations](#99-coordinate-transformations)
  * [9.10. Affine & Curvilinear Transformations](#910-affine---curvilinear-transformations)
- [10. Rank of Matrix](#10-rank-of-matrix)
  * [10.1. Conclusion on Computing Rank](#101-conclusion-on-computing-rank)
- [11. Dimension of the Column Space](#11-dimension-of-the-column-space)
- [12. Null Space (Kernel)](#12-null-space--kernel-)
  * [12.1. Example of Calculating Null Space](#121-example-of-calculating-null-space)
- [13. Nullity](#13-nullity)
- [14. Rank-nullity Theorem](#14-rank-nullity-theorem)
- [15. The Determinant of The Matrix](#15-the-determinant-of-the-matrix)
- [16. Finding The Inverse of The Matrix](#16-finding-the-inverse-of-the-matrix)
- [17. The Fundamental Theorem of Linear Algebra](#17-the-fundamental-theorem-of-linear-algebra)
- [18. Permutation Matrix](#18-permutation-matrix)
- [19. Augmented Matrix](#19-augmented-matrix)


# 1. Vector Space

A vector space is a set <img src="https://latex.codecogs.com/svg.image?V" /> vectors together with two binary operations (vector addition and scalar multiplication) that satisfy the **eight axioms** listed below. In this context, the , and the .


1. Associativity of vector addition: <img src="https://latex.codecogs.com/svg.image?u+%20\left(%20v%20+%20w%20%20\right)%20%20%20=%20\left(%20u%20+%20%20v%20\right)%20+%20w" alt="https://latex.codecogs.com/svg.image?u+ \left( v + w  \right)   = \left( u +  v \right) + w " />
2. Commutativity of vector addition: <img src="https://latex.codecogs.com/svg.image?u%20+%20v%20=%20v%20+%20u" alt="https://latex.codecogs.com/svg.image?u + v = v + u " />

3. Identity element of vector addition: There exists an element <img src="https://latex.codecogs.com/svg.image?0%20%20\in%20V" alt="https://latex.codecogs.com/svg.image?0 \in V" />, called the zero vector, such that<img src="https://latex.codecogs.com/svg.image?0%20%20+%20\mathbf{v}=\mathbf{v}" alt="https://latex.codecogs.com/svg.image?0  + \mathbf{v}=\mathbf{v} " /> for all <img src="https://latex.codecogs.com/svg.image?\mathbf{v}%20%20\in%20V" alt="https://latex.codecogs.com/svg.image?\mathbf{v}  \in V " />.

4. Inverse elements of vector addition: For every <img src="https://latex.codecogs.com/svg.image?\mathbf{v}%20%20\in%20V" alt="https://latex.codecogs.com/svg.image?\mathbf{v}  \in V " />, there exists an element<img src="https://latex.codecogs.com/svg.image?\mathbf{-v}%20%20\in%20V" alt="https://latex.codecogs.com/svg.image?\mathbf{-v}  \in V " />, called the additive inverse of <img src="https://latex.codecogs.com/svg.image?\mathbf{v}" alt="https://latex.codecogs.com/svg.image?\mathbf{v}" />, such that <img src="https://latex.codecogs.com/svg.image?%20\mathbf{v}+%20\left%20(%20\mathbf{-v}%20%20%20\right%20)%20=0" alt="https://latex.codecogs.com/svg.image? \mathbf{v}+ \left ( \mathbf{-v}   \right ) =0" />. 

5. Compatibility of scalar multiplication with field multiplication: <img src="https://latex.codecogs.com/svg.image?%20a(b%20\mathbf{v})%20=%20(ab)\mathbf{v}" alt="https://latex.codecogs.com/svg.image? a(b \mathbf{v}) = (ab)\mathbf{v}" />

6. Identity element of scalar multiplication: <img src="https://latex.codecogs.com/svg.image?%201%20\mathbf{v}%20=%20\mathbf{v}" alt="https://latex.codecogs.com/svg.image? 1 \mathbf{v} = \mathbf{v}" />, where 1 denotes the multiplicative identity in <img src="https://latex.codecogs.com/svg.image?F" alt="https://latex.codecogs.com/svg.image?F" />.

7. Distributivity of scalar multiplication with respect to vector addition: <img src="https://latex.codecogs.com/svg.image?a%20\left%20(%20\mathbf{u}%20+%20\mathbf{v}%20%20\right%20)=a\mathbf{u}+a\mathbf{v}" alt="https://latex.codecogs.com/svg.image?a%20\left%20(%20\mathbf{u}%20+%20\mathbf{v}%20%20\right%20)=a\mathbf{u}+a\mathbf{v}" />  

8. Distributivity of scalar multiplication with respect to field addition: <img src="https://latex.codecogs.com/svg.image?\left%20(%20a+b%20\right%20)%20\mathbf{v}%20=a\mathbf{v}+b\mathbf{v}" alt="https://latex.codecogs.com/svg.image?\left ( a+b \right ) \mathbf{v} =a\mathbf{v}+b\mathbf{v}" />









## 1.1. Examples of Vector Spaces

1. Trivial or zero vector space
The simplest example of a vector space is the trivial one: <img src="https://latex.codecogs.com/svg.image?\{0\}" />, which contains only the zero vector (third axiom in the Vector space)

2. Coordinate space


## 1.2. Vector Products
### 1.2.1 Dot Product

### 1.2.2 The Hadamard product (Schur product)
we use <img  src="https://latex.codecogs.com/svg.latex?s%20%5Codot%20t" alt="https://latex.codecogs.com/svg.latex?s \odot t" /> to denote the element wise product of the two vectors.

<img  src="https://latex.codecogs.com/svg.latex?%28s%20%5Codot%20t%29_j%20%3D%20s_j%20t_j" alt="https://latex.codecogs.com/svg.latex?https://latex.codecogs.com/svg.latex?%28s%20%5Codot%20t%29_j%20%3D%20s_j%20t_j" />


<img  src="https://latex.codecogs.com/svg.latex?%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7D%201%20%5C%5C%202%20%5Cend%7Barray%7D%5Cright%5D%20%5Codot%20%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7D%203%20%5C%5C%204%5Cend%7Barray%7D%20%5Cright%5D%20%3D%20%5Cleft%5B%20%5Cbegin%7Barray%7D%7Bc%7D%201%20*%203%20%5C%5C%202%20*%204%20%5Cend%7Barray%7D%20%5Cright%5D%20%3D%20%5Cleft%5B%20%5Cbegin%7Barray%7D%7Bc%7D%203%20%5C%5C%208%20%5Cend%7Barray%7D%20%5Cright%5D." alt="https://latex.codecogs.com/svg.latex?\left[\begin{array}{c} 1 \\ 2 \end{array}\right]   \odot \left[\begin{array}{c} 3 \\ 4\end{array} \right] = \left[ \begin{array}{c} 1 * 3 \\ 2 * 4 \end{array} \right] = \left[ \begin{array}{c} 3 \\ 8 \end{array}\right]." />






### 1.2.3 Kronecker product
If <img  src="https://latex.codecogs.com/svg.latex?A" alt="https://latex.codecogs.com/svg.latex?A" /> is an <img  src="https://latex.codecogs.com/svg.latex?m%20%5Ctimes%20n" alt="https://latex.codecogs.com/svg.latex?m \times n" /> matrix and <img  src="https://latex.codecogs.com/svg.latex?B" alt="https://latex.codecogs.com/svg.latex?B" /> is a <img  src="https://latex.codecogs.com/svg.latex?p%20%5Ctimes%20q" alt="https://latex.codecogs.com/svg.latex?p \times q" /> matrix, then the Kronecker product <img  src="https://latex.codecogs.com/svg.latex?A%20%5Cbigotimes%20B" alt="https://latex.codecogs.com/svg.latex?A \bigotimes B " /> is the <img  src="https://latex.codecogs.com/svg.latex?pm%20%5Ctimes%20qn" alt="https://latex.codecogs.com/svg.latex?pm \times qn" /> block matrix:

<img  src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%7B%5Cbegin%7Bbmatrix%7D1%262%5C%5C3%264%5C%5C%5Cend%7Bbmatrix%7D%7D%5Cotimes%20%7B%5Cbegin%7Bbmatrix%7D0%265%5C%5C6%267%5C%5C%5Cend%7Bbmatrix%7D%7D%3D%7B%5Cbegin%7Bbmatrix%7D1%7B%5Cbegin%7Bbmatrix%7D0%265%5C%5C6%267%5C%5C%5Cend%7Bbmatrix%7D%7D%262%7B%5Cbegin%7Bbmatrix%7D0%265%5C%5C6%267%5C%5C%5Cend%7Bbmatrix%7D%7D%5C%5C3%7B%5Cbegin%7Bbmatrix%7D0%265%5C%5C6%267%5C%5C%5Cend%7Bbmatrix%7D%7D%264%7B%5Cbegin%7Bbmatrix%7D0%265%5C%5C6%267%5C%5C%5Cend%7Bbmatrix%7D%7D%5C%5C%5Cend%7Bbmatrix%7D%7D%3D%7B%5Cbegin%7Bbmatrix%7D1%5Ctimes%200%261%5Ctimes%205%262%5Ctimes%200%262%5Ctimes%205%5C%5C1%5Ctimes%206%261%5Ctimes%207%262%5Ctimes%206%262%5Ctimes%207%5C%5C3%5Ctimes%200%263%5Ctimes%205%264%5Ctimes%200%264%5Ctimes%205%5C%5C3%5Ctimes%206%263%5Ctimes%207%264%5Ctimes%206%264%5Ctimes%207%5C%5C%5Cend%7Bbmatrix%7D%7D%3D%7B%5Cbegin%7Bbmatrix%7D0%265%260%2610%5C%5C6%267%2612%2614%5C%5C0%2615%260%2620%5C%5C18%2621%2624%2628%5Cend%7Bbmatrix%7D%7D.%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{bmatrix}1&2\\3&4\\\end{bmatrix}}\otimes {\begin{bmatrix}0&5\\6&7\\\end{bmatrix}}={\begin{bmatrix}1{\begin{bmatrix}0&5\\6&7\\\end{bmatrix}}&2{\begin{bmatrix}0&5\\6&7\\\end{bmatrix}}\\3{\begin{bmatrix}0&5\\6&7\\\end{bmatrix}}&4{\begin{bmatrix}0&5\\6&7\\\end{bmatrix}}\\\end{bmatrix}}={\begin{bmatrix}1\times 0&1\times 5&2\times 0&2\times 5\\1\times 6&1\times 7&2\times 6&2\times 7\\3\times 0&3\times 5&4\times 0&4\times 5\\3\times 6&3\times 7&4\times 6&4\times 7\\\end{bmatrix}}={\begin{bmatrix}0&5&0&10\\6&7&12&14\\0&15&0&20\\18&21&24&28\end{bmatrix}}.}" />






# 2. Linear Equation

In many applications we have a system of equations

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}a_{11}x_{1}+a_{12}x_{2}+\cdots%20+a_{1n}x_{n}&=b_{1}\\a_{21}x_{1}+a_{22}x_{2}+\cdots%20+a_{2n}x_{n}&=b_{2}\\&\%20\%20\vdots%20\\a_{m1}x_{1}+a_{m2}x_{2}+\cdots%20+a_{mn}x_{n}&=b_{m}\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}a_{11}x_{1}+a_{12}x_{2}+\cdots +a_{1n}x_{n}&=b_{1}\\a_{21}x_{1}+a_{22}x_{2}+\cdots +a_{2n}x_{n}&=b_{2}\\&\ \ \vdots \\a_{m1}x_{1}+a_{m2}x_{2}+\cdots +a_{mn}x_{n}&=b_{m}\end{aligned}}}" /> 


<br>
<br>

Which can be written as a single matrix equation:

<img src="https://latex.codecogs.com/svg.latex?%20\begin{pmatrix}%20a_{11}&%20%20%20a_{12}&%20...%20&%20a_{1n}\\%20a_{21}&%20%20%20a_{22}&%20...%20&%20a_{2n}\\%20\vdots%20%20%20&%20%20\vdots%20&%20%20&\vdots%20\\%20a_{m1}&%20%20%20a_{m2}&%20...%20&%20a_{mn}\\%20\end{pmatrix}%20\begin{pmatrix}%20x_{1}%20\\%20x_{2}%20\\%20\vdots%20\\%20x_{n}%20\\%20\end{pmatrix}=%20\begin{pmatrix}%20b_{1}%20\\%20b_{2}%20\\%20\vdots%20\\%20b_{m}%20\\%20\end{pmatrix}"  alt="https://latex.codecogs.com/svg.latex? 
\begin{pmatrix}
 a_{11}&   a_{12}& ... & a_{1n}\\ 
 a_{21}&   a_{22}& ... & a_{2n}\\ 
 \vdots   &  \vdots &  &\vdots \\ 
 a_{m1}&   a_{m2}& ... & a_{mn}\\ 
\end{pmatrix}
\begin{pmatrix}
x_{1} \\ 
x_{2} \\ 
\vdots \\ 
x_{n} \\ 
\end{pmatrix}=
\begin{pmatrix}
b_{1} \\ 
b_{2} \\ 
\vdots \\ 
b_{m} \\ 
\end{pmatrix}" /> 
<br>
<br>
Or:
<br>
<br>
<img src="https://latex.codecogs.com/svg.latex?A_{m\times%20n}x_{n\times%201}=b_{m%20\times%201}" alt="https://latex.codecogs.com/svg.latex?A_{m\times n}x_{n\times 1}=b_{m \times 1}" /> 
<br>
<br>

## 2.1. Intuition behind Matrix Multiplication:
We can interpret matrix multiplication as the linear combination of columns:

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}a%20&%20%20b\\c%20&%20%20d\\\end{bmatrix}\begin{bmatrix}x%20\\y\end{bmatrix}=x\begin{bmatrix}a%20\\c\end{bmatrix}+y\begin{bmatrix}%20b\\d\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix} a &  b\\c &  d\\\end{bmatrix}\begin{bmatrix} x \\y\end{bmatrix}=x\begin{bmatrix}a \\c\end{bmatrix}+y\begin{bmatrix} b\\d\end{bmatrix}" />

For instance:

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}3%20&%20%200\\0%20&%20%202\\\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix}3 &  0\\ 0 &  2\\ \end{bmatrix}" />


will transform the <img src="https://latex.codecogs.com/svg.image?\hat{i}" alt="https://latex.codecogs.com/svg.image?\hat{i}"  /> into 

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}3%20\\0\\\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix}3%20\\0\\\end{bmatrix}" /> 

and will transform <img src="https://latex.codecogs.com/svg.image?\hat{j}" alt="https://latex.codecogs.com/svg.image?\hat{j}" /> into 


<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}0%20\\2\\\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix}0%20\\2\\\end{bmatrix}" />





## 2.2. Solution set 

A linear system may behave in any one of three possible ways:

- The system has infinitely many solutions.
- The system has a single unique solution.
- The system has no solution.


The answer of a linear system is determined by the relationship between the number of equations and the number of unknowns.

## 2.3. Underdetermined System
a system with fewer equations than unknowns has infinitely many solutions, but it may have no solution. Such a system is known as an underdetermined system.

## 2.4. Overdetermined System
A system with more equations than unknowns is called as an overdetermined system.

## 2.5. Determined
A system with the same number of equations and unknowns.



Depending on what your matrices looks like, you can choose between various decompositions, and depending on whether you favor speed or accuracy.

## 2.6. Homogeneous vs Non-homogeneous 
A system of linear equations is homogeneous if all of the constant terms are zero.

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}a_{11}x_{1}+a_{12}x_{2}+\cdots%20+a_{1n}x_{n}&=0\\a_{21}x_{1}+a_{22}x_{2}+\cdots%20+a_{2n}x_{n}&=0\\&\%20\%20\vdots%20\\a_{m1}x_{1}+a_{m2}x_{2}+\cdots%20+a_{mn}x_{n}&=0\end{aligned}}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}a_{11}x_{1}+a_{12}x_{2}+\cdots +a_{1n}x_{n}&=0\\a_{21}x_{1}+a_{22}x_{2}+\cdots +a_{2n}x_{n}&=0\\&\ \ \vdots \\a_{m1}x_{1}+a_{m2}x_{2}+\cdots +a_{mn}x_{n}&=0\end{aligned}}}" /> 


# 3. Solving Linear Equation

## 3.1. Using the SVD

If need to solve the least squares problem,(but are not interested in the SVD), a **faster** alternative method is **CompleteOrthogonalDecomposition**. 


```cpp
#include <iostream>
#include <Eigen/Dense>

Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
std::cout << "Here is the matrix A:\n" << A << std::endl;
Eigen::VectorXf b = Eigen::VectorXf::Random(3);
std::cout << "Here is the right hand side b:\n" << b << std::endl;
std::cout << "The least-squares solution is:\n"
 << A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) << std::endl;
```


## 3.2. Complete Orthogonal Decomposition


This class performs a rank-revealing complete orthogonal decomposition of a matrix `A` into matrices `P, Q, T`, and `Z` such that

<img src="https://latex.codecogs.com/svg.image?\mathbf{A}%20\,%20\mathbf{P}%20=%20\mathbf{Q}%20\,%20\begin{bmatrix}%20\mathbf{T}%20&%20\mathbf{0}%20\\%20\mathbf{0}%20&%20\mathbf{0}%20\end{bmatrix}%20\,%20\mathbf{Z}" alt="https://latex.codecogs.com/svg.image?\mathbf{A} \, \mathbf{P} = \mathbf{Q} \, \begin{bmatrix} \mathbf{T} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} \end{bmatrix} \, \mathbf{Z}" />


- `P` is a permutation matrix.
- `Q` and `Z` are unitary matrices and `T` an upper triangular matrix of size rank-by-rank. 
- `A` may be rank deficient.



## 3.3. Using the QR
The `solve()` method in QR decomposition classes also computes the least squares solution. There are three QR decomposition classes: 
1. `HouseholderQR` (no pivoting, **fast** but **unstable** if your matrix is not rull rank), 
2. `ColPivHouseholderQR` (column pivoting, thus a bit **slower** but **more stable**) 
3. `FullPivHouseholderQR` (full pivoting, so **slowest** and slightly **more stable** than `ColPivHouseholderQR`).

```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
Eigen::VectorXf b = Eigen::VectorXf::Random(3);
cout << "The solution using the QR decomposition is:\n"
     << A.colPivHouseholderQr().solve(b) << endl;
```

## 3.4.  Using Cholesky Decomposition

solution of <img src="https://latex.codecogs.com/svg.image?Ax=b" alt="https://latex.codecogs.com/svg.image?Ax=b" /> is equivalent to solving the normal equation <img src="https://latex.codecogs.com/svg.image?A^TAx=A^Tb" alt="https://latex.codecogs.com/svg.image?A^TAx=A^Tb"  />

This method is usually the fastest, especially when A is **tall and skinny**. However, if the matrix A is even mildly ill-conditioned, this is not a good method, because the condition number of <img src="https://latex.codecogs.com/svg.image?A^TA" alt="https://latex.codecogs.com/svg.image?A^TA" /> is the square of the condition number of A. This means that you lose roughly twice as many digits of accuracy using the normal equation, compared to the more stable methods mentioned above.

<img src="https://latex.codecogs.com/svg.image?A%20=%20P^TLDL^*P"  alt="https://latex.codecogs.com/svg.image?A=P^TLDL^*P" />

```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
Eigen::VectorXf b = Eigen::VectorXf::Random(3);
std::cout << "The solution using normal equations is:\n" << (A.transpose() * A).ldlt().solve(A.transpose() * b) << std::endl;
```

Refs: [1](https://eigen.tuxfamily.org/dox-devel/group__LeastSquares.html)

## 3.5. Gaussian Elimination (row reduction)
Gaussian Elimination (row reduction) can be used to solve the systems of linear equations. 
It consists of a sequence of elementary row operations to modify the matrix until the lower left-hand corner of the matrix is filled with zeros and turn into row echelon form . 


There are three types of elementary row operations:

1. Swapping two rows,
2. Multiplying a row by a nonzero number,
3. Adding a multiple of one row to another row.


This method can also be used to compute 
- The rank of a matrix.
- The determinant of a square matrix.
- Inverse of an invertible matrix.

### 3.5.1. Forward Elimination
### 3.5.2. Forward and Back Substitution

A matrix equation in the form <img src="https://latex.codecogs.com/svg.image?{\displaystyle%20L\mathbf%20{x}%20=\mathbf%20{b}%20}" alt="https://latex.codecogs.com/svg.image?{\displaystyle L\mathbf {x} =\mathbf {b} }" /> or <img src="https://latex.codecogs.com/svg.image?{\displaystyle%20U\mathbf%20{x}%20=\mathbf%20{b}%20}" alt="https://latex.codecogs.com/svg.image?{\displaystyle U\mathbf {x} =\mathbf {b} }" /> is very easy to solve.


In lower triangular you first compute <img src="https://latex.codecogs.com/svg.image?x_{1}" alt="https://latex.codecogs.com/svg.image?x_{1}" /> then substitutes that forward into the next equation to solve for <img src="https://latex.codecogs.com/svg.image?x_{2}" alt="https://latex.codecogs.com/svg.image?x_{2}" /> and repeats through to 
<img src="https://latex.codecogs.com/svg.image?x_{n}" alt="https://latex.codecogs.com/svg.image?x_{n}" />
This is called **Forward Substitution**

In an upper triangular matrix, one works backwards, first computing 
<img src="https://latex.codecogs.com/svg.image?x_{n}" alt="https://latex.codecogs.com/svg.image?x_{n}" /> then substituting that back into the previous equation to solve for <img src="https://latex.codecogs.com/svg.image?x_{n-1}" alt="https://latex.codecogs.com/svg.image?x_{n-1}" /> and repeating through <img src="https://latex.codecogs.com/svg.image?x_{1}" alt="https://latex.codecogs.com/svg.image?x_{1}" />. This is called **Back Substitution**.



### 3.5.3. Partial Pivoting and Full Pivoting
- Partial pivoting is about changing the rows of the matrix, effectively changing the order of the equations, for the case when  the pivot is zero and and also for the case when the pivot is a very small number so might lose accuracy due to the round off error.
 
- Full pivoting means both row and column interchanges, for instance we find the biggest element in the matrix and we swap rows and columns untill it becomes the most left-top element for pivoting. This is usually doen for more numerical stability. 

Refs: [1](https://www.youtube.com/watch?v=S5dL9xOj0lU&list=PLkZjai-2Jcxn35XnijUtqqEg0Wi5Sn8ab&index=25)
### 3.5.4. Numerical stability in Gaussian Elimination
In Gaussian elimination it is generally desirable to choose a pivot element with large absolute value. For instance in the following matrix:

<img src="https://latex.codecogs.com/svg.image?\left[{\begin{array}{cc|c}0.00300&59.14&59.17\\5.291&-6.130&46.78\\\end{array}}\right]"
alt="\left[{\begin{array}{cc|c}0.00300&59.14&59.17\\5.291&-6.130&46.78\\\end{array}}\right]" />


The solution is `x1 = 10.00` and `x2 = 1.000`, but when the elimination algorithm   performed with four-digit arithmetic, the small value of <img src="https://latex.codecogs.com/svg.image?a_{11}" />  yields the approximation of `x1 ≈ 9873.3` and `x2 ≈ 4`.

In this case we should interchange the two rows so that <img src="https://latex.codecogs.com/svg.image?a_{21}" /> is in the pivot position 

<img src="https://latex.codecogs.com/svg.image?\left[{\begin{array}{cc|c}5.291&-6.130&46.78\\0.00300&59.14&59.17\\\end{array}}\right]." alt="\left[{\begin{array}{cc|c}5.291&-6.130&46.78\\0.00300&59.14&59.17\\\end{array}}\right]." />


### 3.5.5. Example of The Gaussian Elimination Algorithm

Suppose the following system of linear equations:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{alignedat}2x&{}+{}&y&{}-{}&z&{}={}&8&\qquad%20(L_{1})\\-3x&{}-{}&y&{}+{}&2z&{}={}&-11&\qquad%20(L_{2})\\-2x&{}+{}&y&{}+{}&2z&{}={}&-3&\qquad%20(L_{3})\end{alignedat}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{alignedat}2x&{}+{}&y&{}-{}&z&{}={}&8&\qquad (L_{1})\\-3x&{}-{}&y&{}+{}&2z&{}={}&-11&\qquad (L_{2})\\-2x&{}+{}&y&{}+{}&2z&{}={}&-3&\qquad (L_{3})\end{alignedat}}}" />

Augmented matrix:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left[{\begin{array}{rrr|r}2&1&-1&8\\-3&-1&2&-11\\-2&1&2&-3\end{array}}\right]}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left[{\begin{array}{rrr|r}2&1&-1&8\\-3&-1&2&-11\\-2&1&2&-3\end{array}}\right]}" />


<br/>
<br/>




<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}L_{2}+{\tfrac%20{3}{2}}L_{1}&\to%20L_{2}\\L_{3}+L_{1}&\to%20L_{3}\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}L_{2}+{\tfrac {3}{2}}L_{1}&\to L_{2}\\L_{3}+L_{1}&\to L_{3}\end{aligned}}}" />

<br/>
<br/>

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left[{\begin{array}{rrr|r}2&1&-1&8\\0&{\frac%20{1}{2}}&{\frac%20{1}{2}}&1\\0&2&1&5\end{array}}\right]}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left[{\begin{array}{rrr|r}2&1&-1&8\\0&{\frac {1}{2}}&{\frac {1}{2}}&1\\0&2&1&5\end{array}}\right]}" />

<br/>
<br/>


<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20L_{3}+-4L_{2}\to%20L_{3}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle L_{3}+-4L_{2}\to L_{3}}" />

<br/>
<br/>

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left[{\begin{array}{rrr|r}2&1&-1&8\\0&{\frac%20{1}{2}}&{\frac%20{1}{2}}&1\\0&0&-1&1\end{array}}\right]}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left[{\begin{array}{rrr|r}2&1&-1&8\\0&{\frac {1}{2}}&{\frac {1}{2}}&1\\0&0&-1&1\end{array}}\right]}" />

<br/>
<br/>

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}L_{2}+{\tfrac%20{1}{2}}L_{3}&\to%20L_{2}\\L_{1}-L_{3}&\to%20L_{1}\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}L_{2}+{\tfrac {1}{2}}L_{3}&\to L_{2}\\L_{1}-L_{3}&\to L_{1}\end{aligned}}}" />

<br/>
<br/>

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left[{\begin{array}{rrr|r}2&1&0&7\\0&{\frac%20{1}{2}}&0&{\frac%20{3}{2}}\\0&0&-1&1\end{array}}\right]}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left[{\begin{array}{rrr|r}2&1&0&7\\0&{\frac {1}{2}}&0&{\frac {3}{2}}\\0&0&-1&1\end{array}}\right]}" />

<br/>
<br/>

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}2L_{2}&\to%20L_{2}\\-L_{3}&\to%20L_{3}\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}2L_{2}&\to L_{2}\\-L_{3}&\to L_{3}\end{aligned}}}" />

<br/>
<br/>

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left[{\begin{array}{rrr|r}2&1&0&7\\0&1&0&3\\0&0&1&-1\end{array}}\right]}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left[{\begin{array}{rrr|r}2&1&0&7\\0&1&0&3\\0&0&1&-1\end{array}}\right]}" />

<br/>
<br/>


<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}L_{1}-L_{2}&\to%20L_{1}\\{\tfrac%20{1}{2}}L_{1}&\to%20L_{1}\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}L_{1}-L_{2}&\to L_{1}\\{\tfrac {1}{2}}L_{1}&\to L_{1}\end{aligned}}}" />


<br/>
<br/>

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left[{\begin{array}{rrr|r}1&0&0&2\\0&1&0&3\\0&0&1&-1\end{array}}\right]}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left[{\begin{array}{rrr|r}1&0&0&2\\0&1&0&3\\0&0&1&-1\end{array}}\right]}" />
<br/>
<br/>

###  3.5.6.  Row Echelon Form

A matrix is in echelon form after a Gaussian elimination process and:

- All rows consisting of only zeroes are at the bottom.
- The left-most nonzero entry (leading entry also called the **pivot**) of every nonzero row is to the right of the leading entry of every row above. 


The following matrix is in row echelon form, but not in reduced row echelon 

<img src="images/row_echelon_form.svg" alt="{\displaystyle \left[{\begin{array}{ccccc}1&a_{0}&a_{1}&a_{2}&a_{3}\\0&0&2&a_{4}&a_{5}\\0&0&0&1&a_{6}\\0&0&0&0&0\end{array}}\right]}"  >



The matrix: 

<img  src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}1&4&7\\0&2&3\end{bmatrix}"  alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}1&2&3\\0&4&5\end{bpmatrix}" />

is echelon, but not triangular (because not square). 

The matrix: 

<img  src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}1&4&7\\0&0&2\\0&0&4\end{bmatrix}"  alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}1&2&3\\0&0&4\\0&0&5\end{bmatrix}" />

is triangular, but not echelon because the leading entry 4 is not to the right of the leading entry 2.
For non-singular square matrices, "row echelon" and "upper triangular" are equivalent.

###  3.5.7.  Reduced Row Echelon Form
A matrix is reduced row echelon form if it is in row echelon form and:
- The leading entry in each nonzero row is a 1 (called a leading 1).
- Each column containing a leading 1 has zeros in all its other entries.


This matrix is in reduced row echelon form, which shows that the left part of the matrix is not always an identity matrix:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left[{\begin{array}{ccccc}1&0&a_{1}&0&b_{1}\\0&1&a_{2}&0&b_{2}\\0&0&0&1&b_{3}\end{array}}\right]}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left[{\begin{array}{ccccc}1&0&a_{1}&0&b_{1}\\0&1&a_{2}&0&b_{2}\\0&0&0&1&b_{3}\end{array}}\right]}" />


echelon form is not **unique**, but every matrix has a unique **reduced row echelon form**.


###  3.5.8.  Example of Row Echelon
<img  src="images/ref0.svg" alt="\begin{bmatrix}
1 & 4 & 2 & 3\\ 
2 & 8 & 2 & 5\\ 
0 & 0 & 1 & 1
\end{bmatrix}" />


### 3.5.9. Echelon Pivot Column
If a matrix is in row-echelon form, then the first nonzero entry of each row is called a pivot, and the columns in which pivots appear are called pivot columns.



# 4. Matrices Decompositions
Depending on what your matrices looks like, you can choose between various decompositions, and depending on whether you favor speed or accuracy.

##  4.1. QR Decomposition
### 4.1.1. Square Matrix QR Decomposition
If <img src="https://latex.codecogs.com/svg.latex?A" />  is a real square matrix, then it may be decomposed as:

<img src="https://latex.codecogs.com/svg.latex?A=QR" /> 

<br>
<br>


Where where <img src="https://latex.codecogs.com/svg.latex?Q" /> is an orthogonal matrix, 
meaning: 
<img src="https://latex.codecogs.com/svg.latex?Q^{T}=Q^{-1}" />
 and <img src="https://latex.codecogs.com/svg.latex?R" /> is an upper triangular matrix.
Furthermore, if <img src="https://latex.codecogs.com/svg.latex?A" /> is invertible, then the factorization is unique if we require the diagonal elements of <img src="https://latex.codecogs.com/svg.latex?R" />  to be positive.


For complex square matrices, <img src="https://latex.codecogs.com/svg.latex?Q" />  is a unitary matrix, meaning 
<img src="https://latex.codecogs.com/svg.latex?Q^{*}=Q^{-1}" />

### 4.1.2.  Rectangular Matrix QR Decomposition
If <img src="https://latex.codecogs.com/svg.latex?A_{m\times%20n}" alt="https://latex.codecogs.com/svg.latex?A_{m\times n}" /> where <img src="https://latex.codecogs.com/svg.latex?%20m%20\geq%20%20n" alt="https://latex.codecogs.com/svg.latex? m \geq  n" /> we can factor it into <img src="https://latex.codecogs.com/svg.latex?m\times%20m" alt="https://latex.codecogs.com/svg.latex?m\times m" /> unitary matrix <img src="https://latex.codecogs.com/svg.latex?Q" /> and  an <img src="https://latex.codecogs.com/svg.latex?m\times%20n" alt="https://latex.codecogs.com/svg.latex?m\times n" /> upper triangular matrix <img src="https://latex.codecogs.com/svg.latex?R" />. Since after <img src="https://latex.codecogs.com/svg.latex?\left%20(m-n%20\right%20)_{th}" alt="https://latex.codecogs.com/svg.latex?\left (m-n \right )_{th}" /> row, in <img src="https://latex.codecogs.com/svg.latex?R" /> all elements are entirely zeroes, we can rewrite our equation in the following form:

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20A_{m\times%20n}=Q%20_{m\times%20m}%20%20R_{m\times%20n}%20=Q{\begin{bmatrix}R_{1}\\0\end{bmatrix}}={\begin{bmatrix}Q_{1}&Q_{2}\end{bmatrix}}{\begin{bmatrix}R_{1}\\0\end{bmatrix}}=Q_{1}R_{1},}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle A_{m\times n}=Q _{m\times m}  R_{m\times n} =Q{\begin{bmatrix}R_{1}\\0\end{bmatrix}}={\begin{bmatrix}Q_{1}&Q_{2}\end{bmatrix}}{\begin{bmatrix}R_{1}\\0\end{bmatrix}}=Q_{1}R_{1},}" />


where 
<img src="https://latex.codecogs.com/svg.latex?R_1" /> is an <img src="https://latex.codecogs.com/svg.latex?n\times%20n" alt="https://latex.codecogs.com/svg.latex?n\times n" /> upper triangular matrix and <img src="https://latex.codecogs.com/svg.latex?Q_1" /> is <img src="https://latex.codecogs.com/svg.latex?%20m%20\times%20n" alt="https://latex.codecogs.com/svg.latex? m \times n" />  with orthogonal columns



## 4.2. Computing the QR Decomposition
## 4.3. Gram Schmidt Orthogonalization 
Gram–Schmidt process is a method for orthonormalizing a set of vectors. In this process you make every column perpendicular to it's previous columns. Lets first define the **projection operator** by

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\mathrm%20{proj}%20_{\mathbf%20{u}%20}(\mathbf%20{v}%20)={\frac%20{\langle%20\mathbf%20{u}%20,\mathbf%20{v}%20\rangle%20}{\langle%20\mathbf%20{u}%20,\mathbf%20{u}%20\rangle%20}}{\mathbf%20{u}%20}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \mathrm {proj} _{\mathbf {u} }(\mathbf {v} )={\frac {\langle \mathbf {u} ,\mathbf {v} \rangle }{\langle \mathbf {u} ,\mathbf {u} \rangle }}{\mathbf {u} }}" />

where <img src="https://latex.codecogs.com/svg.latex?\langle%20\mathbf{u},%20\mathbf{v}\rangle" alt="https://latex.codecogs.com/svg.latex?\langle \mathbf{u}, \mathbf{v}\rangle"> denotes the inner product.


Explanation, Let's put 

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\mathrm%20{proj}%20_{\mathbf%20{u}%20}(\mathbf%20{v}%20)}=\vec%20{\mathbf%20{v\prime}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \mathrm {proj} _{\mathbf {u} }(\mathbf {v} )}=\vec {\mathbf {v\prime}} ">

So we already know:


<img src="https://latex.codecogs.com/svg.latex?cos(\theta)=\frac{%20|\vec{\mathbf%20{v\prime}}|%20}{|\vec{\mathbf%20{v}}|}" alt="https://latex.codecogs.com/svg.latex?cos(\theta)=\frac{ |\vec{\mathbf {v\prime}}| }{|\vec{\mathbf {v}}|}">




which mean the magnitude of projection vector is:

<img src="https://latex.codecogs.com/svg.latex?|\vec{\mathbf%20v}|cos(\theta)=%20|\vec%20{\mathbf%20{{v}\prime}}|" alt="https://latex.codecogs.com/svg.latex?|\vec{\mathbf v}|cos(\theta)= |\vec {\mathbf {{v}\prime}}|">




and the direction is in the direction <img src="https://latex.codecogs.com/svg.latex?\mathbf%20{u}" alt="https://latex.codecogs.com/svg.latex?\mathbf {u}">, so you have to make a unit vector of <img src="https://latex.codecogs.com/svg.latex?\mathbf%20{u}" alt="https://latex.codecogs.com/svg.latex?\mathbf {u}"> (by dividing it by its size) and then multiply by the magnitude of the projection vector that we just calculated. 

so putting everything together:


<img src="https://latex.codecogs.com/svg.latex?|\vec{\mathbf%20v}|cos(\theta)=%20|\vec%20{\mathbf%20{{v}\prime}}|"  alt="https://latex.codecogs.com/svg.latex?\vec{\mathbf v}.\vec{\mathbf u}= |\vec{\mathbf v}||\vec{\mathbf u}|cos(\theta)" />

<br/>
<br/>



<img src="https://latex.codecogs.com/svg.latex?\vec%20{\mathbf{v\prime}}=|\vec%20{\mathbf{v\prime}}|%20%20\frac{\vec%20{\mathbf{u}}%20%20}{|\vec%20{\mathbf{u}}%20|%20}" alt="https://latex.codecogs.com/svg.latex?\vec {\mathbf{v\prime}}=|\vec {\mathbf{v\prime}}|  \frac{\vec {\mathbf{u}}  }{|\vec {\mathbf{u}} | }" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?\vec{\mathbf%20v}.\vec{\mathbf%20u}=%20|\vec{\mathbf%20v}||\vec{\mathbf%20u}|cos(\theta)" alt="https://latex.codecogs.com/svg.latex?\vec {\mathbf{v\prime}}=|\vec {\mathbf{v\prime}}|  \frac{\vec {\mathbf{u}}  }{|\vec {\mathbf{u}} | } ">

will give us:


<img src="https://latex.codecogs.com/svg.latex?\vec%20{\mathbf{v\prime}}=|\vec%20{\mathbf{v\prime}}|%20%20\frac{\vec%20{\mathbf{u}}%20%20}{|\vec%20{\mathbf{u}}%20|%20}%20=|\vec{\mathbf%20v}|cos(\theta)\frac{\vec%20{\mathbf{u}}%20%20}{|\vec%20{\mathbf{u}}%20|%20}=|\vec%20{\mathbf{u}}%20||\vec{\mathbf%20v}|cos(\theta)\frac{\vec%20{\mathbf{u}}%20%20}{|\vec%20{\mathbf{u}}%20||\vec%20{\mathbf{u}}%20|%20}" alt="https://latex.codecogs.com/svg.latex?\vec {\mathbf{v\prime}}=|\vec {\mathbf{v\prime}}|  \frac{\vec {\mathbf{u}}  }{|\vec {\mathbf{u}} | } =|\vec{\mathbf v}|cos(\theta)\frac{\vec {\mathbf{u}}  }{|\vec {\mathbf{u}} | }=|\vec {\mathbf{u}} ||\vec{\mathbf v}|cos(\theta)\frac{\vec {\mathbf{u}}  }{|\vec {\mathbf{u}} ||\vec {\mathbf{u}} | }">

<br/>
<br/>





Now lets imagine we have the following vectors,  

<img src="images/gram_schmidt1.png">


The Gram–Schmidt process has the followings steps:

<img src="images/gram_schmidt2.png">

<br>


<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}\mathbf%20{u}%20_{1}&=\mathbf%20{v}%20_{1},&\mathbf%20{e}%20_{1}&={\frac%20{\mathbf%20{u}%20_{1}}{\|\mathbf%20{u}%20_{1}\|}}\end{aligned}}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}\mathbf {u} _{1}&=\mathbf {v} _{1},&\mathbf {e} _{1}&={\frac {\mathbf {u} _{1}}{\|\mathbf {u} _{1}\|}}\end{aligned}}}">

<br>


<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}%20\\\mathbf%20{u}%20_{2}&=\mathbf%20{v}%20_{2}-\mathrm%20{proj}%20_{\mathbf%20{u}%20_{1}}(\mathbf%20{v}%20_{2}),&\mathbf%20{e}%20_{2}&={\frac%20{\mathbf%20{u}%20_{2}}{\|\mathbf%20{u}%20_{2}\|}}%20\end{aligned}}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned} \\\mathbf {u} _{2}&=\mathbf {v} _{2}-\mathrm {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{2}),&\mathbf {e} _{2}&={\frac {\mathbf {u} _{2}}{\|\mathbf {u} _{2}\|}} \end{aligned}}}">
<br>

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}%20\\\mathbf%20{u}%20_{3}&=\mathbf%20{v}%20_{3}-\mathrm%20{proj}%20_{\mathbf%20{u}%20_{1}}(\mathbf%20{v}%20_{3})-\mathrm%20{proj}%20_{\mathbf%20{u}%20_{2}}(\mathbf%20{v}%20_{3}),&\mathbf%20{e}%20_{3}&={\frac%20{\mathbf%20{u}%20_{3}}{\|\mathbf%20{u}%20_{3}\|}}%20\end{aligned}}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned} 
\\\mathbf {u} _{3}&=\mathbf {v} _{3}-\mathrm {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{3})-\mathrm {proj} _{\mathbf {u} _{2}}(\mathbf {v} _{3}),&\mathbf {e} _{3}&={\frac {\mathbf {u} _{3}}{\|\mathbf {u} _{3}\|}}
 \end{aligned}}}">
<br>




<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}%20\\\mathbf%20{u}%20_{4}&=\mathbf%20{v}%20_{4}-\mathrm%20{proj}%20_{\mathbf%20{u}%20_{1}}(\mathbf%20{v}%20_{4})-\mathrm%20{proj}%20_{\mathbf%20{u}%20_{2}}(\mathbf%20{v}%20_{4})-\mathrm%20{proj}%20_{\mathbf%20{u}%20_{3}}(\mathbf%20{v}%20_{4}),&\mathbf%20{e}%20_{4}&={\mathbf%20{u}%20_{4}%20\over%20\|\mathbf%20{u}%20_{4}\|}\\&{}\%20\%20\vdots%20&&{}\%20\%20\vdots%20\end{aligned}}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}
\\\mathbf {u} _{4}&=\mathbf {v} _{4}-\mathrm {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{4})-\mathrm {proj} _{\mathbf {u} _{2}}(\mathbf {v} _{4})-\mathrm {proj} _{\mathbf {u} _{3}}(\mathbf {v} _{4}),&\mathbf {e} _{4}&={\mathbf {u} _{4} \over \|\mathbf {u} _{4}\|}\\&{}\ \ \vdots &&{}\ \ \vdots
 \end{aligned}}}">
 <br>

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\mathbf%20{u}%20_{k}=\mathbf%20{v}%20_{k}-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{1}}(\mathbf%20{v}%20_{k})-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{2}}(\mathbf%20{v}%20_{k})-\cdots%20-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{k-1}}(\mathbf%20{v}%20_{k})}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \mathbf {u} _{k}=\mathbf {v} _{k}-\operatorname {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{k})-\operatorname {proj} _{\mathbf {u} _{2}}(\mathbf {v} _{k})-\cdots -\operatorname {proj} _{\mathbf {u} _{k-1}}(\mathbf {v} _{k})}"/>

<br>

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}%20\\\mathbf%20{u}%20_{k}&=\mathbf%20{v}%20_{k}-\sum%20_{j=1}^{k-1}\mathrm%20{proj}%20_{\mathbf%20{u}%20_{j}}(\mathbf%20{v}%20_{k}),&\mathbf%20{e}%20_{k}&={\frac%20{\mathbf%20{u}%20_{k}}{\|\mathbf%20{u}%20_{k}\|}}.%20\end{aligned}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned} 
\\\mathbf {u} _{k}&=\mathbf {v} _{k}-\sum _{j=1}^{k-1}\mathrm {proj} _{\mathbf {u} _{j}}(\mathbf {v} _{k}),&\mathbf {e} _{k}&={\frac {\mathbf {u} _{k}}{\|\mathbf {u} _{k}\|}}.
\end{aligned}}}">

<br>
<br>

<img src="images/gram_schmidt3.png">
<br>
<br>



<img src="images/gram_schmidt4.png">
<br>


<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}%20\\\mathbf%20{u}%20_{2}&=\mathbf%20{v}%20_{2}-\mathrm%20{proj}%20_{\mathbf%20{u}%20_{1}}(\mathbf%20{v}%20_{2}),&\mathbf%20{e}%20_{2}&={\frac%20{\mathbf%20{u}%20_{2}}{\|\mathbf%20{u}%20_{2}\|}}%20\end{aligned}}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned} \\\mathbf {u} _{2}&=\mathbf {v} _{2}-\mathrm {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{2}),&\mathbf {e} _{2}&={\frac {\mathbf {u} _{2}}{\|\mathbf {u} _{2}\|}} \end{aligned}}}">

<br>
<br>


Due to rounding errors, the vectors <img src="https://latex.codecogs.com/svg.latex?\mathbf%20{u}_{k}" alt="https://latex.codecogs.com/svg.latex?\mathbf {u}_{k}"/> are often not quite orthogonal, therefore, it is said that the (classical) Gram–Schmidt process is numerically unstable. This can be stabilized by a small modification, where 
Instead of computing the vector <img src="https://latex.codecogs.com/svg.latex?\mathbf%20{u}_{k}" alt="https://latex.codecogs.com/svg.latex?\mathbf {u}_{k}"/> as:

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\mathbf%20{u}%20_{k}=\mathbf%20{v}%20_{k}-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{1}}(\mathbf%20{v}%20_{k})-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{2}}(\mathbf%20{v}%20_{k})-\cdots%20-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{k-1}}(\mathbf%20{v}%20_{k})}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \mathbf {u} _{k}=\mathbf {v} _{k}-\operatorname {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{k})-\operatorname {proj} _{\mathbf {u} _{2}}(\mathbf {v} _{k})-\cdots -\operatorname {proj} _{\mathbf {u} _{k-1}}(\mathbf {v} _{k})}"/>

<br>
<br>
We do the following:
<br>
<br>


<img src="https://latex.codecogs.com/svg.latex?%20{\displaystyle%20{\begin{aligned}\mathbf%20{u}%20_{k}^{(1)}&=\mathbf%20{v}%20_{k}-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{1}}(\mathbf%20{v}%20_{k}),\\\mathbf%20{u}%20_{k}^{(2)}&=\mathbf%20{u}%20_{k}^{(1)}-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{2}}\left(\mathbf%20{u}%20_{k}^{(1)}\right),\\&\;\;\vdots%20\\\mathbf%20{u}%20_{k}^{(k-2)}&=\mathbf%20{u}%20_{k}^{(k-3)}-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{k-2}}\left(\mathbf%20{u}%20_{k}^{(k-3)}\right),\\\mathbf%20{u}%20_{k}^{(k-1)}&=\mathbf%20{u}%20_{k}^{(k-2)}-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{k-1}}\left(\mathbf%20{u}%20_{k}^{(k-2)}\right),\\\mathbf%20{u}%20_{k}&={\frac%20{\mathbf%20{u}%20_{k}^{(k-1)}}{\left\|\mathbf%20{u}%20_{k}^{(k-1)}\right\|}}\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?
{\displaystyle {\begin{aligned}\mathbf {u} _{k}^{(1)}&=\mathbf {v} _{k}-\operatorname {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{k}),\\\mathbf {u} _{k}^{(2)}&=\mathbf {u} _{k}^{(1)}-\operatorname {proj} _{\mathbf {u} _{2}}\left(\mathbf {u} _{k}^{(1)}\right),\\&\;\;\vdots \\\mathbf {u} _{k}^{(k-2)}&=\mathbf {u} _{k}^{(k-3)}-\operatorname {proj} _{\mathbf {u} _{k-2}}\left(\mathbf {u} _{k}^{(k-3)}\right),\\\mathbf {u} _{k}^{(k-1)}&=\mathbf {u} _{k}^{(k-2)}-\operatorname {proj} _{\mathbf {u} _{k-1}}\left(\mathbf {u} _{k}^{(k-2)}\right),\\\mathbf {u} _{k}&={\frac {\mathbf {u} _{k}^{(k-1)}}{\left\|\mathbf {u} _{k}^{(k-1)}\right\|}}\end{aligned}}}" 
/>




<img src="images/gram_schmidt5.png" />
<br>
<br>
<img src="images/gram_schmidt6.png" />
<br>
<br>


<img alt="https://latex.codecogs.com/svg.latex?{v}_{3}^{'}=   \mathbf {u}_{3}^{(1)}" src="https://latex.codecogs.com/svg.latex?{v}_{3}^{%27}=%20%20%20\mathbf%20{u}_{3}^{(1)}"/>


<img src="https://latex.codecogs.com/svg.latex?%20{\displaystyle%20{\begin{aligned}%20\mathbf%20{u}_{3}^{(1)}&=\mathbf%20{v}%20_{3}-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{1}}(\mathbf%20{v}%20_{3})\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?
{\displaystyle {\begin{aligned} \mathbf {u}_{3}^{(1)}&=\mathbf {v} _{3}-\operatorname {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{3})\end{aligned}}}" />





<img src="images/gram_schmidt7.png" />
<br>
<br>

<img alt="https://latex.codecogs.com/svg.latex?\mathbf{u}_{3}=   \mathbf {u}_{3}^{(2)}" src="https://latex.codecogs.com/svg.latex?\mathbf{u}_{3}=%20%20%20\mathbf%20{u}_{3}^{(2)}"/>


<img src="https://latex.codecogs.com/svg.latex?%20{\displaystyle%20{\begin{aligned}%20\mathbf%20{u}_{3}^{(2)}&=\mathbf%20{u}^{(1)}%20_{3}-\operatorname%20{proj}%20_{\mathbf%20{u}%20_{2}}(\mathbf%20{u}^{(1)}%20_{3})\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?
{\displaystyle {\begin{aligned} \mathbf {u}_{3}^{(2)}&=\mathbf {u}^{(1)} _{3}-\operatorname {proj} _{\mathbf {u} _{2}}(\mathbf {u}^{(1)} _{3})\end{aligned}}}" />

We can now express the  s over our newly computed orthonormal basis:


<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}\mathbf%20{v}%20_{1}&=\left\langle%20\mathbf%20{e}%20_{1},\mathbf%20{v}%20_{1}\right\rangle%20\mathbf%20{e}%20_{1}\\\mathbf%20{v}%20_{2}&=\left\langle%20\mathbf%20{e}%20_{1},\mathbf%20{v}%20_{2}\right\rangle%20\mathbf%20{e}%20_{1}+\left\langle%20\mathbf%20{e}%20_{2},\mathbf%20{v}%20_{2}\right\rangle%20\mathbf%20{e}%20_{2}\\\mathbf%20{v}%20_{3}&=\left\langle%20\mathbf%20{e}%20_{1},\mathbf%20{v}%20_{3}\right\rangle%20\mathbf%20{e}%20_{1}+\left\langle%20\mathbf%20{e}%20_{2},\mathbf%20{v}%20_{3}\right\rangle%20\mathbf%20{e}%20_{2}+\left\langle%20\mathbf%20{e}%20_{3},\mathbf%20{v}%20_{3}\right\rangle%20\mathbf%20{e}%20_{3}\\&\;\;\vdots%20\\\mathbf%20{v}%20_{k}&=\sum%20_{j=1}^{k}\left\langle%20\mathbf%20{e}%20_{j},\mathbf%20{v}%20_{k}\right\rangle%20\mathbf%20{e}%20_{j}\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}\mathbf {v} _{1}&=\left\langle \mathbf {e} _{1},\mathbf {v} _{1}\right\rangle \mathbf {e} _{1}\\\mathbf {v} _{2}&=\left\langle \mathbf {e} _{1},\mathbf {v} _{2}\right\rangle \mathbf {e} _{1}+\left\langle \mathbf {e} _{2},\mathbf {v} _{2}\right\rangle \mathbf {e} _{2}\\\mathbf {v} _{3}&=\left\langle \mathbf {e} _{1},\mathbf {v} _{3}\right\rangle \mathbf {e} _{1}+\left\langle \mathbf {e} _{2},\mathbf {v} _{3}\right\rangle \mathbf {e} _{2}+\left\langle \mathbf {e} _{3},\mathbf {v} _{3}\right\rangle \mathbf {e} _{3}\\&\;\;\vdots \\\mathbf {v} _{k}&=\sum _{j=1}^{k}\left\langle \mathbf {e} _{j},\mathbf {v} _{k}\right\rangle \mathbf {e} _{j}\end{aligned}}}" />
<br>
<br>



This can be written in matrix form:
<br>
<br>

<img src="https://latex.codecogs.com/svg.latex?A%20=%20QR" alt="https://latex.codecogs.com/svg.latex?A = QR" />

<br>
<br>




<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20Q={\begin{bmatrix}\mathbf%20{e}%20_{1}&\cdots%20&\mathbf%20{e}%20_{n}\end{bmatrix}}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle Q={\begin{bmatrix}\mathbf {e} _{1}&\cdots &\mathbf {e} _{n}\end{bmatrix}}}" />

<br>
<br>


<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20R={\begin{bmatrix}\langle%20\mathbf%20{e}%20_{1},\mathbf%20{v}%20_{1}\rangle%20&\langle%20\mathbf%20{e}%20_{1},\mathbf%20{v}%20_{2}\rangle%20&\langle%20\mathbf%20{e}%20_{1},\mathbf%20{v}%20_{3}\rangle%20&\cdots%20\\0&\langle%20\mathbf%20{e}%20_{2},\mathbf%20{v}%20_{2}\rangle%20&\langle%20\mathbf%20{e}%20_{2},\mathbf%20{v}%20_{3}\rangle%20&\cdots%20\\0&0&\langle%20\mathbf%20{e}%20_{3},\mathbf%20{v}%20_{3}\rangle%20&\cdots%20\\\vdots%20&\vdots%20&\vdots%20&\ddots%20\end{bmatrix}}.}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle R={\begin{bmatrix}\langle \mathbf {e} _{1},\mathbf {v} _{1}\rangle &\langle \mathbf {e} _{1},\mathbf {v} _{2}\rangle &\langle \mathbf {e} _{1},\mathbf {v} _{3}\rangle &\cdots \\0&\langle \mathbf {e} _{2},\mathbf {v} _{2}\rangle &\langle \mathbf {e} _{2},\mathbf {v} _{3}\rangle &\cdots \\0&0&\langle \mathbf {e} _{3},\mathbf {v} _{3}\rangle &\cdots \\\vdots &\vdots &\vdots &\ddots \end{bmatrix}}.}" />



## 4.4. Householder Transformations


Refs: [1](https://www.youtube.com/watch?v=pOiOH3yESPM)

## 4.5. QL, RQ and LQ Decompositions
We can define <img src="https://latex.codecogs.com/svg.latex?QL" />, <img src="https://latex.codecogs.com/svg.latex?RQ" />, and <img src="https://latex.codecogs.com/svg.latex?LQ" /> decompositions, with <img src="https://latex.codecogs.com/svg.latex?L" /> being a lower triangular matrix.

## 4.6. Cholesky Decomposition LL*

Cholesky decomposition is a decomposition of a Hermitian, positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose


## 4.7. Hermitian Matrix
Hermitian Matrix means a matrix that its transpose is its conjugate:




<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20{\begin{bmatrix}0&a-ib&c-id\\a+ib&1&m-in\\c+id&m+in&2\end{bmatrix}}}"  alt="https://latex.codecogs.com/svg.image?{\displaystyle {\begin{bmatrix}0&a-ib&c-id\\a+ib&1&m-in\\c+id&m+in&2\end{bmatrix}}}" />



## 4.8. Positive (Semidefinite) Definite Matrix:
Matrix <img src="https://latex.codecogs.com/svg.image?M_{n\times%20n}" alt="https://latex.codecogs.com/svg.image?M_{n\times n}"  /> is said to be positive definite if for every the nonzero real column vector <img src="https://latex.codecogs.com/svg.image?z_{n\times%201}" alt="https://latex.codecogs.com/svg.image?z_{n\times 1}" /> the scalar 
<img src="https://latex.codecogs.com/svg.image?z^T%20M%20z" alt="https://latex.codecogs.com/svg.image?z^TMz" />
is positive.


<img src="https://latex.codecogs.com/svg.image?\text{A%20Positive%20Semidefinite}%20\Leftrightarrow%20\text{All%20eigenvalues%20}%20\lambda%20\geq%200" alt="https://latex.codecogs.com/svg.image?\text{A Positive Semidefinite} \Leftrightarrow \text{All eigenvalues } \lambda \geq 0" />





Example:

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}2%20&%20-1%20&%200%20\\-1%20&%202%20&%20-1%20\\0%20&%20-1%20&%202%20\\\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \\ \end{bmatrix}"   />


<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}a\\b\\c\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix}a\\b\\c\end{bmatrix}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?z^TMz=a^2%20+c^2%20+%20(a-b)^2%20+(b-c)^2" alt="https://latex.codecogs.com/svg.image?z^TMz=a^2 +c^2 + (a-b)^2 +(b-c)^2" />

Cholesky decomposition of a **Hermitian positive-definite** matrix A is:
  
<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbf%20{A}%20=\mathbf%20{LL}%20^{*},}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbf {A} =\mathbf {LL} ^{*},}" />  


  
- <img src="https://latex.codecogs.com/svg.image?L" alt="https://latex.codecogs.com/svg.image?L" />  is a lower triangular matrix with real and positive diagonal entries.
- <img src="https://latex.codecogs.com/svg.image?L^*" alt="https://latex.codecogs.com/svg.image?L^*" /> is the conjugate transpose of  <img src="https://latex.codecogs.com/svg.image?L" alt="https://latex.codecogs.com/svg.image?L" /> 


## 4.9. LDL Decomposition
A closely related variant of the classical Cholesky decomposition is the LDL decomposition:

<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbf%20{A}%20=\mathbf%20{LDL}%20^{*}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbf {A} =\mathbf {LDL} ^{*}}" />

where <img src="https://latex.codecogs.com/svg.image?L" alt="https://latex.codecogs.com/svg.image?L" /> is a lower unit triangular (unitriangular) matrix, and <img src="https://latex.codecogs.com/svg.image?D" alt="https://latex.codecogs.com/svg.image?D" /> is a diagonal matrix


Example:

<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbf%20{A}%20=\mathbf%20{LL}%20^{*}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbf {A} =\mathbf {LL} ^{*}}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20{\begin{aligned}\left({\begin{array}{*{3}{r}}4&12&-16\\12&37&-43\\-16&-43&98\\\end{array}}\right)=\left({\begin{array}{*{3}{r}}2&0&0\\6&1&0\\-8&5&3\\\end{array}}\right)\left({\begin{array}{*{3}{r}}2&6&-8\\0&1&5\\0&0&3\\\end{array}}\right).\end{aligned}}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle%20{\begin{aligned}\left({\begin{array}{*{3}{r}}4&12&-16\\12&37&-43\\-16&-43&98\\\end{array}}\right)=\left({\begin{array}{*{3}{r}}2&0&0\\6&1&0\\-8&5&3\\\end{array}}\right)\left({\begin{array}{*{3}{r}}2&6&-8\\0&1&5\\0&0&3\\\end{array}}\right).\end{aligned}}}" />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbf%20{A}%20=\mathbf%20{LDL}%20^{*}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbf {A} =\mathbf {LDL} ^{*}}" />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20{\begin{aligned}\left({\begin{array}{*{3}{r}}4&12&-16\\12&37&-43\\-16&-43&98\\\end{array}}\right)&=\left({\begin{array}{*{3}{r}}1&0&0\\3&1&0\\-4&5&1\\\end{array}}\right)\left({\begin{array}{*{3}{r}}4&0&0\\0&1&0\\0&0&9\\\end{array}}\right)\left({\begin{array}{*{3}{r}}1&3&-4\\0&1&5\\0&0&1\\\end{array}}\right).\end{aligned}}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle%20{\begin{aligned}\left({\begin{array}{*{3}{r}}4&12&-16\\12&37&-43\\-16&-43&98\\\end{array}}\right)&=\left({\begin{array}{*{3}{r}}1&0&0\\3&1&0\\-4&5&1\\\end{array}}\right)\left({\begin{array}{*{3}{r}}4&0&0\\0&1&0\\0&0&9\\\end{array}}\right)\left({\begin{array}{*{3}{r}}1&3&-4\\0&1&5\\0&0&1\\\end{array}}\right).\end{aligned}}}" />

Refs: [1](https://www.youtube.com/watch?v=2uKoKKLgZ4c)

## 4.10. Lower Upper (LU) Decomposition

Lower-upper (LU) decomposition factors a matrix as the product of a lower triangular matrix and an upper triangular matrix
LU decomposition can be viewed as the matrix form of Gaussian elimination.

In the lower triangular matrix all elements above the diagonal are zero, in the upper triangular matrix, all the elements below the diagonal are zero.

<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20{\begin{bmatrix}a_{11}&a_{12}&a_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{bmatrix}}={\begin{bmatrix}\ell%20_{11}&0&0\\\ell%20_{21}&\ell%20_{22}&0\\\ell%20_{31}&\ell%20_{32}&\ell%20_{33}\end{bmatrix}}{\begin{bmatrix}u_{11}&u_{12}&u_{13}\\0&u_{22}&u_{23}\\0&0&u_{33}\end{bmatrix}}.}" alt="https://latex.codecogs.com/svg.image?{\displaystyle {\begin{bmatrix}a_{11}&a_{12}&a_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{bmatrix}}={\begin{bmatrix}\ell _{11}&0&0\\\ell _{21}&\ell _{22}&0\\\ell _{31}&\ell _{32}&\ell _{33}\end{bmatrix}}{\begin{bmatrix}u_{11}&u_{12}&u_{13}\\0&u_{22}&u_{23}\\0&0&u_{33}\end{bmatrix}}.}" />


## 4.11. Lower Diagonal Upper (LDU) decomposition
A Lower-diagonal-upper (LDU) decomposition is a decomposition of the form

<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20A=LDU}" alt="https://latex.codecogs.com/svg.image?{\displaystyle%20A=LDU}" />

where <img src="https://latex.codecogs.com/svg.image?D" alt="https://latex.codecogs.com/svg.image?D" /> is a diagonal matrix, and <img src="https://latex.codecogs.com/svg.image?L" alt="https://latex.codecogs.com/svg.image?L" />  and <img src="https://latex.codecogs.com/svg.image?U" alt="https://latex.codecogs.com/svg.image?U" />  are unitriangular matrices, meaning that all the entries on the diagonals of <img src="https://latex.codecogs.com/svg.image?L" alt="https://latex.codecogs.com/svg.image?L" />  and <img src="https://latex.codecogs.com/svg.image?U" alt="https://latex.codecogs.com/svg.image?U" />  are one.


## 4.12. Eigen Value and Eigen Vector


An eigenvalue and eigenvector are a scalar value and a non-zero vector that, when a linear transformation is applied to it, changes only by a scalar factor. 
More formally, if <img src="https://latex.codecogs.com/svg.image?T" alt="https://latex.codecogs.com/svg.image?T" /> is a linear transformation and <img src="https://latex.codecogs.com/svg.image?v" alt="https://latex.codecogs.com/svg.image?v" /> is a vector, then

<img src="https://latex.codecogs.com/svg.image?T(v)%20=%20\lambda%20v"  alt="https://latex.codecogs.com/svg.image?T(v)=\lambda v" />

## 4.13. Calculation of Eigen Value and Eigen Vector

where <img src="https://latex.codecogs.com/svg.image?\lambda" alt="https://latex.codecogs.com/svg.image?\lambda" /> is a scalar (the eigenvalue) and <img src="https://latex.codecogs.com/svg.image?v" alt="https://latex.codecogs.com/svg.image?v" /> is the eigenvector. 

<img src="https://latex.codecogs.com/svg.image?Av=\lambda%20v" alt="https://latex.codecogs.com/svg.image?Av=\lambda v" />

<br/>

<img src="https://latex.codecogs.com/svg.image?(A-\lambda%20I)v=0" alt="https://latex.codecogs.com/svg.image?(A-\lambda I)v=0" />

<br/>

<img src="https://latex.codecogs.com/svg.image?det(A-\lambda%20I)=0" alt="https://latex.codecogs.com/svg.image?det(A-\lambda I)=0" />


### 4.13.1  Example of Calculating Eigen Value and Eigen Vector

<img src="https://latex.codecogs.com/svg.image?A=\begin{bmatrix}2%20&%200%20&%200%20\\0%20&%204%20&%205%20\\0%20&%204%20&%203%20\\\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?A=\begin{bmatrix} 2 & 0 & 0 \\ 0 & 4 & 5 \\ 0 & 4 & 3 \\ \end{bmatrix}"  />

<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}2%20&%200%20&%200%20\\0%20&%204%20&%205%20\\0%20&%204%20&%203%20\\\end{bmatrix}-\lambda\begin{bmatrix}1%20&%200%20&%200%20\\0%20&%201%20&%200%20\\0%20&%200%20&%201%20\\\end{bmatrix}%20=\begin{bmatrix}2-\lambda%20&%200%20&%200%20\\0%20&%204-\lambda%20&%205%20\\0%20&%204%20&%203-\lambda%20\\\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix} 2 & 0 & 0 \\ 0 & 4 & 5 \\ 0 & 4 & 3 \\ \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix} = \begin{bmatrix} 2-\lambda & 0 & 0 \\ 0 & 4-\lambda & 5 \\ 0 & 4 & 3-\lambda \\ \end{bmatrix}"  />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?\begin{vmatrix}2-\lambda%20&%200%20&%200%20\\0%20&%204-\lambda%20&%205%20\\0%20&%204%20&%203-\lambda%20\\\end{vmatrix}%20=0%20\to%20(2-\lambda)[(4-\lambda)(3-\lambda)%20-5\times%204]" alt="https://latex.codecogs.com/svg.image?\begin{vmatrix} 2-\lambda & 0 & 0 \\ 0 & 4-\lambda & 5 \\ 0 & 4 & 3-\lambda \\ \end{vmatrix} =0 \to  (2-\lambda)[(4-\lambda)(3-\lambda) -5\times 4]"  />

<br/>
<br/>

Eigenvalues are −1, 2 and 8.

<img src="" alt=""  />

<img src="" alt=""  />

## 4.14. Eigendecomposition of Matrix

## 4.15. Singular Value Decomposition
Singular value decomposition (SVD) is a factorization of a real or (complex matrix) which generalizes the eigendecomposition of a square matrix. 

<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5C%20%5Cmathbf%20%7BM_%7Bm%5Ctimes%20n%7D%7D%20%3D%5Cmathbf%20%7BU_%7Bm%5Ctimes%20m%7D%20%5CSigma_%7Bm%5Ctimes%20n%7D%20V%5E%7B*%7D_%7Bn%5Ctimes%20n%7D%7D%20%5C%20%2C%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \ \mathbf {M_{m\times n}} =\mathbf {U_{m\times m} \Sigma_{m\times n} V^{*}_{n\times n}} \ ,}" />

1. <img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%20%7BU_%7Bm%5Ctimes%20m%7D%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf {U_{m\times m}}"  /> is complex unitary matrix


2. <img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5C%20%5Cmathbf%20%7B%5CSigma_%7Bm%5Ctimes%20n%7D%20%7D%20%5C%20%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \ \mathbf {\Sigma_{m\times n} } \ } "  /> rectangular diagonal matrix with non-negative real numbers on the diagonal.  
3.  The diagonal entries <img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5C%20%5Csigma%20_%7Bi%7D%3D%5CSigma%20_%7Bii%7D%5C%20%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \ \sigma _{i}=\Sigma _{ii}\ }"  /> of  <img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5C%20%5Cmathbf%20%7B%5CSigma%20%7D%20%5C%20%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \ \mathbf {\Sigma } \ }"  /> are **uniquely** determined by `M` and are known as the singular values of M. 

4. <img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5C%20%5Cmathbf%20%7BV%5E%7B*%7D%7D%20%5C%20%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \ \mathbf {V^{*}} \ }"  />  is the conjugate transpose of `V`. Such decomposition always exists for any complex matrix. 

5. If `M`is real, then `U` and `V` can be guaranteed to be real orthogonal matrices.


6. The columns of `U` <img src="https://latex.codecogs.com/svg.latex?u_1%2C%20...%2C%20u_n" alt="https://latex.codecogs.com/svg.latex?u_1, ..., u_n"  /> and the columns of `V` <img src="https://latex.codecogs.com/svg.latex?v_1%2C%20...%2C%20v_n" alt="https://latex.codecogs.com/svg.latex?v_1, ..., v_n"  /> are form two sets of orthonormal bases and the singular value decomposition can be written as:

<br/>
 <img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5C%20%5Cmathbf%20%7BM%7D%20%3D%5Csum%20_%7Bi%3D1%7D%5E%7Br%7D%5Csigma%20_%7Bi%7D%5Cmathbf%20%7Bu%7D%20_%7Bi%7D%5Cmathbf%20%7Bv%7D%20_%7Bi%7D%5E%7B*%7D%5C%20%2C%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \ \mathbf {M} =\sum _{i=1}^{r}\sigma _{i}\mathbf {u} _{i}\mathbf {v} _{i}^{*}\ ,}"  /> 

<br/> 
 
<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5C%20r%5Cleq%20%5Cmin%5C%7Bm%2Cn%5C%7D%5C%20%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \ r\leq \min\{m,n\}\ } "  /> 


7. The SVD is **not** unique.


<img src="images/Singular-Value-Decomposition.svg" alt="" width="75%" height="75%" />



###  4.15.1 Applications of the SVD
#### 4.15.1.1 Pseudoinverse


<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%7B%5Cdisplaystyle%20%5C%20%5Cmathbf%20%7BM%7D%20%3D%5Cmathbf%20%7BU%5CSigma%20V%5E%7B*%7D%7D%20%5C%20%2C%7D%20%5C%5C%20%7B%5Cdisplaystyle%20%5C%20%5Cmathbf%20%7BM%5E%7B%5Cdagger%7D%20%7D%20%3D%5Cmathbf%20%7BV%5CSigma%5E%7B%5Cdagger%7D%20U%5E%7B*%7D%7D%20%5C%20%2C%7D" alt="https://latex.codecogs.com/svg.latex?\\
{\displaystyle \ \mathbf {M} =\mathbf {U\Sigma V^{*}} \ ,}
\\
{\displaystyle \ \mathbf {M^{\dagger} } =\mathbf {V\Sigma^{\dagger} U^{*}} \ ,}"  />




<img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%20%7BM%5E%7B%5Cdagger%7D%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf {M^{\dagger}} "  /> is formed by replacing every non-zero diagonal entry by its reciprocal <img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B1%7D%7B%5Csigma_i%7D" alt="https://latex.codecogs.com/svg.latex?\frac{1}{\sigma_i}"  />  and transposing the resulting matrix. 


<br/>

<img src="https://latex.codecogs.com/svg.latex?U%5CSigma%20V%5ET%20x%20%3D%20b%20%5Cimplies%5C%5C%20%5CSigma%20%28V%5ET%20x%20%29%20%3D%20U%5ET%20b%20%5Cimplies%5C%5C%20V%5ET%20x%20%3D%20%5CSigma%5E&plus;%20U%5ET%20b%20%5Cimplies%5C%5C%20x%20%3D%20%28V%20%5CSigma%20%5E&plus;%20U%5ET%29b" alt="https://latex.codecogs.com/svg.latex?U\Sigma V^T x = b \implies\\
\Sigma (V^T x ) = U^T b \implies\\
V^T x = \Sigma^+ U^T b \implies\\
x = (V \Sigma ^+ U^T)b" />

#### 4.15.1.2 Solving homogeneous linear equations

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20Ah%3D0%20%5C%5C%20A%20%3D%20UDV%5ET" alt="https://latex.codecogs.com/svg.latex?\\
Ah=0
\\
A = UDV^T"  />


<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20V_%7Bm%5Ctimes%20m%7D%3D%5Cbegin%7Bbmatrix%7D%20v_1%20%26%20v_2%20%26%20...%20%26%20v_i%20%26%20..%20%26v_m%5C%5C%20%5Cend%7Bbmatrix%7D%20%5C%5C%20V%5ETV%3D%7B%5Cdisplaystyle%20I_%7Bm%7D%3D%7B%5Cbegin%7Bbmatrix%7D1%260%260%26%5Ccdots%20%260%5C%5C0%261%260%26%5Ccdots%20%260%5C%5C0%260%261%26%5Ccdots%20%260%5C%5C%5Cvdots%20%26%5Cvdots%20%26%5Cvdots%20%26%5Cddots%20%26%5Cvdots%20%5C%5C0%260%260%26%5Ccdots%20%261%5Cend%7Bbmatrix%7D%7D_%7Bm%5Ctimes%20m%7D%7D%20%5C%5C%20V%5ETv_i%3De_i%3D%5Cbegin%7Bbmatrix%7D%200%5C%5C%20%5Cvdots%20%5C%5C%201%5C%5C%20%5Cvdots%20%5C%5C%20%5C%5C%200%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\\
V_{m\times m}=\begin{bmatrix}
v_1 & v_2 & ... & v_i & .. &v_m\\ 
\end{bmatrix}
\\
V^TV={\displaystyle  I_{m}={\begin{bmatrix}1&0&0&\cdots &0\\0&1&0&\cdots &0\\0&0&1&\cdots &0\\\vdots &\vdots &\vdots &\ddots &\vdots \\0&0&0&\cdots &1\end{bmatrix}}_{m\times m}}
\\
V^Tv_i=e_i=\begin{bmatrix}
0\\ 
\vdots \\ 
1\\ 
\vdots \\ 
\\ 
0
\end{bmatrix}" />


for any matrix:

<br/>

<img src="https://latex.codecogs.com/svg.latex?Be_i%3Db_i" alt="https://latex.codecogs.com/svg.latex?Be_i=b_i" />

<br/>
<br/>

therefore:

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?Av_i%3DU%20D%20V%5ET%20v_i%20%3D%20U%20D%20e_i%20%3D%20U%20%28%5Csigma_i%20e_i%29%3D%5Csigma_i%20u_i." alt="https://latex.codecogs.com/svg.latex?Av_i=U D V^T v_i = U D e_i = U (\sigma_i e_i)=\sigma_i u_i." />




Refs: [1](https://math.stackexchange.com/questions/1768181/svd-and-homogeneous-equation)


####  4.15.1.3  Range, null space and rank

<img  src="https://latex.codecogs.com/svg.latex?A%20v_i%3D%20%5Csigma_i%20u_i" alt="https://latex.codecogs.com/svg.latex?A v_i= \sigma_i u_i" />

The last columns do serve as a basis for the null space:


<img src="https://latex.codecogs.com/svg.latex?%5C%7Bv_i%7Ci%3Erank%5C%7D" alt="https://latex.codecogs.com/svg.latex?\{v_i|i>rank\}" />



Refs: [1](https://math.stackexchange.com/questions/1771013/how-is-the-null-space-related-to-singular-value-decomposition)

####  4.15.1.4 Nearest orthogonal matrix


<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5Cmathbf%20%7BO%7D%20%3D%7B%5Cunderset%20%7B%5COmega%20%7D%7B%5Coperatorname%20%7Bargmin%7D%20%7D%7D%5C%7C%5Cmathbf%20%7BA%7D%20%7B%5Cboldsymbol%20%7B%5COmega%20%7D%7D-%5Cmathbf%20%7BB%7D%20%5C%7C_%7BF%7D%5Cquad%20%7B%5Ctext%7Bsubject%20to%7D%7D%5Cquad%20%7B%5Cboldsymbol%20%7B%5COmega%20%7D%7D%5E%7B%5Ctextsf%20%7BT%7D%7D%7B%5Cboldsymbol%20%7B%5COmega%20%7D%7D%3D%5Cmathbf%20%7BI%7D%20%2C%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \mathbf {O} ={\underset {\Omega }{\operatorname {argmin} }}\|\mathbf {A} {\boldsymbol {\Omega }}-\mathbf {B} \|_{F}\quad {\text{subject to}}\quad {\boldsymbol {\Omega }}^{\textsf {T}}{\boldsymbol {\Omega }}=\mathbf {I} ,}" />



# 5. Linear Map
Let  <img  src="https://latex.codecogs.com/svg.latex?V"  alt="https://latex.codecogs.com/svg.latex?V" /> and  <img  src="https://latex.codecogs.com/svg.latex?W"  alt="https://latex.codecogs.com/svg.latex?W" /> be vector spaces over the same field  <img  src="https://latex.codecogs.com/svg.latex?K"  alt="https://latex.codecogs.com/svg.latex?K" />. A function <img  src="https://latex.codecogs.com/svg.latex?f:V\to%20W"  alt="https://latex.codecogs.com/svg.latex?f:V\to W" />  is said to be a linear map if for any two vectors
 <img  src="https://latex.codecogs.com/svg.latex?{\textstyle%20\mathbf%20{u}%20,\mathbf%20{v}%20\in%20V}"  alt="https://latex.codecogs.com/svg.latex?{\textstyle \mathbf {u} ,\mathbf {v} \in V}" />  and any scalar 
  <img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20c\in%20K}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle c\in K}" />  the following two conditions are satisfied:

1. Additivity

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20f(\mathbf%20{u}%20+\mathbf%20{v}%20)=f(\mathbf%20{u}%20)+f(\mathbf%20{v}%20)}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle f(\mathbf {u} +\mathbf {v} )=f(\mathbf {u} )+f(\mathbf {v} )}" />


2. Operation of scalar multiplication

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20f(c\mathbf%20{u}%20)=cf(\mathbf%20{u}%20)}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle f(c\mathbf {u} )=cf(\mathbf {u} )}" />




# 6. Span

The linear span of a set <img  src="https://latex.codecogs.com/svg.latex?S"  alt="https://latex.codecogs.com/svg.latex?S" /> of 
vectors for a vector space is as the set of all finite linear combinations of the vectors in <img  src="https://latex.codecogs.com/svg.latex?S"  alt="https://latex.codecogs.com/svg.latex?S" /> 

Example: 
1. The span of 


<img src="https://latex.codecogs.com/svg.image?\left%20(%20%20%20\begin{bmatrix}1\\-2%20\\0\end{bmatrix}%20,%20\begin{bmatrix}3\\1%20\\0\end{bmatrix}\right%20)" alt="\left (   \begin{bmatrix} 1\\-2 \\ 0 \end{bmatrix} ,  \begin{bmatrix} 3\\ 1 \\ 0 \end{bmatrix}" />

is:

<img src="https://latex.codecogs.com/svg.image?v=c_1%20%20%20\begin{bmatrix}1\\-2%20\\0\end{bmatrix}%20+c_2%20\begin{bmatrix}3\\1%20\\0\end{bmatrix}" alt="v=c_1 \begin{bmatrix} 1\\ -2 \\ 0 \end{bmatrix} + c_2  \begin{bmatrix} 3\\ 1 \\ 0 \end{bmatrix}" />

2. The real vector space <img src="https://latex.codecogs.com/svg.image?\mathbb%20{R}%20^{3}" alt="https://latex.codecogs.com/svg.image?\mathbb {R} ^{3}" />  has {(−1, 0, 0), (0, 1, 0), (0, 0, 1)} as a spanning set as a spanning set.


The set {(1, 0, 0), (0, 1, 0), (1, 1, 0)} is **not** a spanning set of 
<img src="https://latex.codecogs.com/svg.image?\mathbb%20{R}%20^{3}" alt="https://latex.codecogs.com/svg.image?\mathbb {R} ^{3}" />, since its span is the space of all vectors in <img src="https://latex.codecogs.com/svg.image?\mathbb%20{R}%20^{3}" alt="https://latex.codecogs.com/svg.image?\mathbb {R} ^{3}" />  whose last component is zero.

# 7. Subspace

A vector subspace is a subset of a vector space that satisfies certain properties, such that:
1. The set includes the zero vector.

2. The set is closed under scalar multiplication.

3. The set is closed under addition.


In other words, if two vectors are in the subspace, their sum and scalar multiples must also be in the subspace. It is also non-empty and closed under linear combinations.


Examples of subspace:



# 7.1. Row Spaces and Column Spaces

The column space of a matrix <img  src="https://latex.codecogs.com/svg.latex?A"  alt="https://latex.codecogs.com/svg.latex?A" /> is the span (set of all possible linear combinations) of its column vectors.

# 8. Range of a Matrix
The range of a matrix, also known as the column space of a matrix, is the span of the columns of the matrix. In other words, it is the set of all possible linear combinations of the columns of the matrix.

```cpp
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
CompleteOrthogonalDecomposition(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M) {
  Eigen::CompleteOrthogonalDecomposition<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      cod(M);
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Q =
      cod.householderQ();
  return Q.leftCols(cod.rank());
}
```



Refs: [1](https://math.stackexchange.com/questions/2037602/what-is-range-of-a-matrix)

## 8.1. Example of Row Spaces

<img src="https://latex.codecogs.com/svg.image?M={\begin{bmatrix}2&4&1&3&2\\-1&-2&1&0&5\\1&6&2&2&2\\3&6&2&5&1\end{bmatrix}}" alt="https://latex.codecogs.com/svg.image?M={\begin{bmatrix}2&4&1&3&2\\-1&-2&1&0&5\\1&6&2&2&2\\3&6&2&5&1\end{bmatrix}}"/>

The rows are:

- <img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbf%20{r}%20_{1}={\begin{bmatrix}2&4&1&3&2\end{bmatrix}}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbf {r} _{1}={\begin{bmatrix}2&4&1&3&2\end{bmatrix}}}" />

- <img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbf%20{r}%20_{2}={\begin{bmatrix}-1&-2&1&0&5\end{bmatrix}}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbf {r} _{2}={\begin{bmatrix}-1&-2&1&0&5\end{bmatrix}}}" />

- <img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbf%20{r}%20_{3}={\begin{bmatrix}1&6&2&2&2\end{bmatrix}}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbf {r} _{3}={\begin{bmatrix}1&6&2&2&2\end{bmatrix}}}" />

- <img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbf%20{r}%20_{4}={\begin{bmatrix}3&6&2&5&1\end{bmatrix}}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbf {r} _{4}={\begin{bmatrix}3&6&2&5&1\end{bmatrix}}}" />

Consequently, the row space of <img src="https://latex.codecogs.com/svg.image?M" alt="https://latex.codecogs.com/svg.image?M" /> is the subspace of <img src="https://latex.codecogs.com/svg.image?{\displaystyle%20\mathbb%20{R}%20^{5}}" alt="https://latex.codecogs.com/svg.image?{\displaystyle \mathbb {R} ^{5}} "  /> spanned by <img src="https://latex.codecogs.com/svg.image?{%20r1,%20r2,%20r3,%20r4%20}"  alt="https://latex.codecogs.com/svg.image?{ r1, r2, r3, r4 }" /> . Since these four row vectors are linearly independent, the row space is 4-dimensional.

<br/>

# 9. Basis


In linear algebra, a basis is a set of linearly independent vectors that can be used to span a vector (sub)space.  The dimension of a vector space is the number of vectors in a basis for the space.
To find the column basis in matrix you have to find the pivot column as they are linearly independent, so first write the matrix in the row echelon form and then pick the pivot columns.

Number of basis for a space is the dimension of that space. The dimension of the column space is the rank of the matrix.

## 9.1. Example of Computing Basis for Column Space

We write the matrix in the row echelon form, and then pick the pivot columns. For example, the matrix A given by

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20A={\begin{bmatrix}1&2&1\\-2&-3&1\\3&5&0\end{bmatrix}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle A={\begin{bmatrix}1&2&1\\-2&-3&1\\3&5&0\end{bmatrix}}}" />


following elementary row operations:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}{\begin{bmatrix}1&2&1\\-2&-3&1\\3&5&0\end{bmatrix}}&\xrightarrow%20{2R_{1}+R_{2}\to%20R_{2}}%20{\begin{bmatrix}1&2&1\\0&1&3\\3&5&0\end{bmatrix}}\xrightarrow%20{-3R_{1}+R_{3}\to%20R_{3}}%20{\begin{bmatrix}1&2&1\\0&1&3\\0&-1&-3\end{bmatrix}}\\&\xrightarrow%20{R_{2}+R_{3}\to%20R_{3}}%20\,\,{\begin{bmatrix}1&2&1\\0&1&3\\0&0&0\end{bmatrix}}\xrightarrow%20{-2R_{2}+R_{1}\to%20R_{1}}%20{\begin{bmatrix}1&0&-5\\0&1&3\\0&0&0\end{bmatrix}}~.\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}{\begin{bmatrix}1&2&1\\-2&-3&1\\3&5&0\end{bmatrix}}&\xrightarrow {2R_{1}+R_{2}\to R_{2}} {\begin{bmatrix}1&2&1\\0&1&3\\3&5&0\end{bmatrix}}\xrightarrow {-3R_{1}+R_{3}\to R_{3}} {\begin{bmatrix}1&2&1\\0&1&3\\0&-1&-3\end{bmatrix}}\\&\xrightarrow {R_{2}+R_{3}\to R_{3}} \,\,{\begin{bmatrix}1&2&1\\0&1&3\\0&0&0\end{bmatrix}}\xrightarrow {-2R_{2}+R_{1}\to R_{1}} {\begin{bmatrix}1&0&-5\\0&1&3\\0&0&0\end{bmatrix}}~.\end{aligned}}}" />


There are two non-zero rows in the final matrix and therefore the rank of matrix is 2 and column 1 and 2 are basis for the column space:



<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}1%20\\%20-2\\3\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix}1 \\-2\\3\end{bmatrix}" /> , <img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}2%20\\%20-3\\5\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix}2 \\-3\\5\end{bmatrix}" />



##  9.2. Example of Computing Basis for Row Space
Let say we have the following matrix:

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}-2%20&&%202%20%20&&%20%206%20&&%20%200\\0%20&&%206%20&&%207%20&&%205\\1%20&&%205%20&&%204%20&&%205\\\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix} -2 && 2  &&  6 &&  0\\ 0 && 6 && 7 && 5\\ 1 && 5 && 4 && 5\\ \end{bmatrix}" />

By writing it into row echelon form:

<img src="https://latex.codecogs.com/svg.image?{\displaystyle%20{\begin{aligned}{\begin{bmatrix}-2&2&6&0\\0&6&7&5%20%20\\1&5&4&5\end{aligned}}&\xrightarrow%20{1/2R_{1}%20+R_{3}%20\to%20R_{3}}%20{\begin{bmatrix}-2&2&6&0\\0&6&7&5%20%20\\0&6&7&5\end{aligned}}\xrightarrow%20{-R_{2}+R_{3}\to%20R_{3}}%20{\begin{bmatrix}-2&2&6&0\\0&6&7&5%20%20\\0&0&0&0\end{bmatrix}}\end{aligned}}}"  alt="{\displaystyle {\begin{aligned}{\begin{bmatrix}-2&2&6&0\\0&6&7&5  \\1&5&4&5\end{aligned}}&\xrightarrow {1/2R_{1} +R_{3} \to R_{3}} {\begin{bmatrix}-2&2&6&0\\0&6&7&5  \\0&6&7&5\end{aligned}}\xrightarrow {-R_{2}+R_{3}\to R_{3}} {\begin{bmatrix}-2&2&6&0\\0&6&7&5  \\0&0&0&0\end{bmatrix}}\end{aligned}}}"  />

Now we pick the non-zero rows, so the basis for row space of our matrix is:


<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}-2%20\\%202\\4%20\\0\end{bmatrix},\begin{bmatrix}0%20\\6%20\\7%20\\5\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix} -2 \\  2\\ 4 \\ 0\end{bmatrix},\begin{bmatrix} 0 \\ 6 \\ 7 \\ 5 \end{bmatrix} ">

##  9.3. Changes of basis vectors
Let say our first basis vector set is:


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%200%5C%5C%201%5Cend%7Bbmatrix%7D," alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} 0\\ 1\end{bmatrix}," /> <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%201%5C%5C%200%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} 1\\ 0\end{bmatrix}" />

and our second basis vector set is:

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%202%5C%5C%201%5Cend%7Bbmatrix%7D%2C%20%5Cbegin%7Bbmatrix%7D%20-1%5C%5C%201%5Cend%7Bbmatrix%7D." alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} 2\\ 1\end{bmatrix}, \begin{bmatrix} -1\\ 1\end{bmatrix}." />

In fact these are the position of the second basis vectors in our first basis set.
If a vector is described as 

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20-1%5C%5C%202%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} -1\\ 2\end{bmatrix}" />

in our second basis we can find it in the first basis as:

<img src="https://latex.codecogs.com/svg.latex?-1%5Ctimes%5Cbegin%7Bbmatrix%7D%202%5C%5C%201%5Cend%7Bbmatrix%7D%20&plus;%202%5Ctimes%5Cbegin%7Bbmatrix%7D%20-1%5C%5C%201%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%20-4%5C%5C%201%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?-1\times\begin{bmatrix} 2\\ 1\end{bmatrix} + 2\times\begin{bmatrix} -1\\ 1\end{bmatrix}=\begin{bmatrix} -4\\ 1\end{bmatrix}" /> 

or matrix multiplication where the columns of the matrix are second basis vectors:

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%202%26%20-1%5C%5C%201%26%201%20%5Cend%7Bbmatrix%7D%20%5Ctimes%20%5Cbegin%7Bbmatrix%7D%20-1%5C%5C%202%5Cend%7Bbmatrix%7D%20%3D%5Cbegin%7Bbmatrix%7D%20-4%5C%5C%201%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}  2& -1\\  1& 1 \end{bmatrix} \times \begin{bmatrix} -1\\ 2\end{bmatrix} =\begin{bmatrix} -4\\ 1\end{bmatrix} " />


##  9.4. Covariance and Contravariance of Vectors


Two ways to describe a vector in basis vectors:

1) Parallel Projection Counting how many unit vectors we should add to get our vector. In our example:

<img src="https://latex.codecogs.com/svg.latex?-1%5Ctimes%20%5Cbegin%7Bbmatrix%7D%202%5C%5C%201%5Cend%7Bbmatrix%7D%20&plus;%202%5Ctimes%5Cbegin%7Bbmatrix%7D%20-1%5C%5C%201%5Cend%7Bbmatrix%7D%3D0" alt="https://latex.codecogs.com/svg.latex?-1\times \begin{bmatrix} 2\\ 1\end{bmatrix} + 2\times\begin{bmatrix} -1\\ 1\end{bmatrix}=0" />

2)perpendicular projection Dot product our vector with basis vector:

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20-1%2C%202%5Cend%7Bbmatrix%7D%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%202%5C%5C1%5Cend%7Bbmatrix%7D%3D0" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} -1, 2\end{bmatrix} \cdot \begin{bmatrix} 2\\1\end{bmatrix}=0" />


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20-1%2C%202%5Cend%7Bbmatrix%7D%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20-1%5C%5C1%5Cend%7Bbmatrix%7D%3D4" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} -1, 2\end{bmatrix} \cdot \begin{bmatrix} -1\\1\end{bmatrix}=4" />


If we double the size of the basis vectors, our new basis is:

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%204%5C%5C%202%5Cend%7Bbmatrix%7D%24%2C%20%24%5Cbegin%7Bbmatrix%7D%20-2%5C%5C2%5Cend%7Bbmatrix%7D." alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} 4\\ 2\end{bmatrix}$, $\begin{bmatrix} -2\\2\end{bmatrix}." />


This will turn our old vector:

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20-1%5C%5C%202%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} -1\\ 2\end{bmatrix}" />

into:

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20-0.5%5C%5C%201%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} -0.5\\ 1\end{bmatrix}" />


Because these two quantities change "contrary" to one other, they are are refereed as "contra-variant" component of the vector.

If we use the second approach to represent our vector, the associated dot product will also double. 


Ref: [1](https://mathinsight.org/taylors_theorem_multivariable_introduction), [2](http://www.math.toronto.edu/courses/mat237y1/20199/notes/Chapter2/S2.6.html), [3](https://www.youtube.com/watch?v=vvE5w3iOtGs), [4](http://jccc-mpg.wikidot.com/vector-projection), [5](https://www.youtube.com/watch?v=P2LTAUO1TdA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=13)

## 9.5. Creating a Basis Set









## 9.6. Change of Basis

The concept of a change of basis in linear algebra involves transitioning from one set of basis vectors to another, effectively redefining how vectors in a space are represented. This is crucial in many areas of mathematics and physics, as it allows for the representation of vectors and linear transformations in the most convenient basis for a given problem.

### The Equation for Change of Basis

Given a vector space <img src="https://latex.codecogs.com/svg.latex?V" alt="V" /> and two bases for this space, say <img src="https://latex.codecogs.com/svg.latex?B%20%3D%20%5C%7Bb_1%2C%20b_2%2C%20...%2C%20b_n%5C%7D" alt="B = \{b_1, b_2, ..., b_n\}" /> and <img src="https://latex.codecogs.com/svg.latex?C%20%3D%20%5C%7Bc_1%2C%20c_2%2C%20...%2C%20c_n%5C%7D" alt="C = \{c_1, c_2, ..., c_n\}" />, a vector <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> in <img src="https://latex.codecogs.com/svg.latex?V" alt="V" /> can be represented in terms of both bases. If <img src="https://latex.codecogs.com/svg.latex?P" alt="P" /> is the change of basis matrix from <img src="https://latex.codecogs.com/svg.latex?B" alt="B" /> to <img src="https://latex.codecogs.com/svg.latex?C" alt="C" />, then the coordinates of <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> in the basis <img src="https://latex.codecogs.com/svg.latex?C" alt="C" /> can be found by multiplying <img src="https://latex.codecogs.com/svg.latex?P" alt="P" /> with the coordinates of <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> in the basis <img src="https://latex.codecogs.com/svg.latex?B" alt="B" />. Mathematically, this is expressed as:

<img src="https://latex.codecogs.com/svg.latex?%5Bv%5D_C%20%3D%20P%20%5Bv%5D_B" alt="[v]_C = P [v]_B" />

Where:
- <img src="https://latex.codecogs.com/svg.latex?%5Bv%5D_B" alt="[v]_B" /> is the representation (coordinates) of <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> in basis <img src="https://latex.codecogs.com/svg.latex?B" alt="B" />,
- <img src="https://latex.codecogs.com/svg.latex?%5Bv%5D_C" alt="[v]_C" /> is the representation (coordinates) of <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> in basis <img src="https://latex.codecogs.com/svg.latex?C" alt="C" />,
- <img src="https://latex.codecogs.com/svg.latex?P" alt="P" /> is the matrix whose columns are the coordinates of the basis vectors of <img src="https://latex.codecogs.com/svg.latex?C" alt="C" /> expressed in the basis <img src="https://latex.codecogs.com/svg.latex?B" alt="B" />.

### Example of Change of Basis

Let's consider a simple 2D example where we change the basis from the standard basis <img src="https://latex.codecogs.com/svg.latex?B%20%3D%20%5C%7B%5Chat%7Bi%7D%2C%20%5Chat%7Bj%7D%5C%7D" alt="B = \{\hat{i}, \hat{j}\}" /> to a new basis <img src="https://latex.codecogs.com/svg.latex?C%20%3D%20%5C%7Bc_1%2C%20c_2%5C%7D" alt="C = \{c_1, c_2\}" />, where <img src="https://latex.codecogs.com/svg.latex?c_1%20%3D%20%281%2C%201%29" alt="c_1 = (1, 1)" /> and <img src="https://latex.codecogs.com/svg.latex?c_2%20%3D%20%281%2C%20-1%29" alt="c_2 = (1, -1)" />.

**Objective**: Find the coordinates of a vector <img src="https://latex.codecogs.com/svg.latex?v%20%3D%20%283%2C%202%29" alt="v = (3, 2)" /> in the new basis <img src="https://latex.codecogs.com/svg.latex?C" alt="C" />.

1. **Express the new basis vectors in terms of the standard basis**:
   - <img src="https://latex.codecogs.com/svg.latex?c_1%20%3D%20%281%2C%201%29" alt="c_1 = (1, 1)" /> corresponds to <img src="https://latex.codecogs.com/svg.latex?1%5Chat%7Bi%7D%20%2B%201%5Chat%7Bj%7D" alt="1\hat{i} + 1\hat{j}" />,
   - <img src="https://latex.codecogs.com/svg.latex?c_2%20%3D%20%281%2C%20-1%29" alt="c_2 = (1, -1)" /> corresponds to <img src="https://latex.codecogs.com/svg.latex?1%5Chat%7Bi%7D%20-%201%5Chat%7Bj%7D" alt="1\hat{i} - 1\hat{j}" />.

2. **Construct the change of basis matrix <img src="https://latex.codecogs.com/svg.latex?P" alt="P" />**:
   - <img src="https://latex.codecogs.com/svg.latex?P%20%3D%20%5Cbegin%7Bpmatrix%7D%201%20%26%201%20%5C%5C%201%20%26%20-1%20%5Cend%7Bpmatrix%7D" alt="P = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}" />, where the columns are <img src="https://latex.codecogs.com/svg.latex?c_1" alt="c_1" /> and <img src="https://latex.codecogs.com/svg.latex?c_2" alt="c_2" /> expressed in the standard basis.

3. **Calculate the coordinates of <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> in the new basis <img src="https://latex.codecogs.com/svg.latex?C" alt="C" />** by solving <img src="https://latex.codecogs.com/svg.latex?%5Bv%5D_C%20%3D%20P%5E%7B-1%7D%20%5Bv%5D_B" alt="[v]_C = P^{-1} [v]_B" />, where <img src="https://latex.codecogs.com/svg.latex?P%5E%7B-1%7D" alt="P^{-1}" /> is the inverse of the change of basis matrix <img src="https://latex.codecogs.com/svg.latex?P" alt="P" />, and <img src="https://latex.codecogs.com/svg.latex?%5Bv%5D_B%20%3D%20%283%2C%202%29" alt="[v]_B = (3, 2)" /> is the representation of <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> in the standard basis.

Let's calculate the coordinates of <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> in the new basis <img src="https://latex.codecogs.com/svg.latex?C" alt="C" />.

The coordinates of the vector <img src="https://latex.codecogs.com/svg.latex?v%20%3D%20%283%2C%202%29" alt="v = (3, 2)" /> in the new basis <img src="https://latex.codecogs.com/svg.latex?C" alt="C" />, where <img src="https://latex.codecogs.com/svg.latex?c_1%20%3D%20%281%2C%201%29" alt="c_1 = (1, 1)" /> and <img src="https://latex.codecogs.com/svg.latex?c_2%20%3D%20%281%2C%20-1%29" alt="c_2 = (1, -1)" />, are <img src="https://latex.codecogs.com/svg.latex?%282.5%2C%200.5%29" alt="(2.5, 0.5)" />. This means that in the basis <img src="https://latex.codecogs.com/svg.latex?C" alt="C" />, the vector <img src="https://latex.codecogs.com/svg.latex?v" alt="v" /> can be represented as <img src="https://latex.codecogs.com/svg.latex?2.5c_1%20%2B%200.5c_2" alt="2.5c_1 + 0.5c_2" />.


## 9.7. Vector Fields
Ref: [1](https://tutorial.math.lamar.edu/classes/calciii/VectorFields.aspx)

## 9.8. Coordinate System
### 9.8.1. Cartesian, Polar, Curvilinear coordinates ,Cylindrical and Spherical Coordinates

Ref: [1](https://www.skillsyouneed.com/num/polar-cylindrical-spherical-coordinates.html)

## 9.9. Coordinate transformations
Refs [1](https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations)

## 9.10. Affine & Curvilinear Transformations



# 10. Rank of Matrix

Let <img  src="https://latex.codecogs.com/svg.latex?A"  alt="https://latex.codecogs.com/svg.latex?A" /> be an m-by-n matrix. Then

- <img src="https://latex.codecogs.com/svg.image?rank(A)"  alt="https://latex.codecogs.com/svg.image?rank(A)" /> = number of pivots in any echelon form of <img  src="https://latex.codecogs.com/svg.latex?A"  alt="https://latex.codecogs.com/svg.latex?A" />

- <img src="https://latex.codecogs.com/svg.image?rank(A)"  alt="https://latex.codecogs.com/svg.image?rank(A)" /> = the maximum number of linearly independent **rows** or **columns** of <img  src="https://latex.codecogs.com/svg.latex?A"  alt="https://latex.codecogs.com/svg.latex?A" />

- <img src="https://latex.codecogs.com/svg.image?rank(A)%20=%20dim(rowsp(A))%20=%20dim(colsp(A))"  alt="https://latex.codecogs.com/svg.image?rank(A) = dim(rowsp(A)) = dim(colsp(A))" />

## 10.1. Conclusion on Computing Rank
In practice, due to floating point error on computers,  Gaussian elimination (LU decomposition) can be unreliable, therefore rank-revealing decomposition such as RRQR factorization (rank-revealing QR which is QR decomposition with pivoting) should be used. The singular value decomposition (SVD) can be used, but it is not an efficient method to do so.



# 11. Dimension of the Column Space

The dimension of the column space or row space is called the rank of the matrix, and is the maximum number of linearly independent columns


# 12. Null Space (Kernel)


If <img  src="https://latex.codecogs.com/svg.latex?A"  alt="https://latex.codecogs.com/svg.latex?A" /> is a matrix, the null-space (The kernel of <img  src="https://latex.codecogs.com/svg.latex?A"  alt="https://latex.codecogs.com/svg.latex?A" /> ) consists of all the linear combinations of vectors that get mapped to the zero vector when multiplied by <img  src="https://latex.codecogs.com/svg.latex?A"  alt="https://latex.codecogs.com/svg.latex?A" />, in other words it is, the set of all vectors <img  src="https://latex.codecogs.com/svg.latex?v"  alt="https://latex.codecogs.com/svg.latex?v" /> such that <img  src="https://latex.codecogs.com/svg.latex?A.v=0"  alt="https://latex.codecogs.com/svg.latex?A.v=0" />. It's good to think of the matrix as a linear transformation; if you let <img  src="https://latex.codecogs.com/svg.latex?h(v)=A.v"  alt="https://latex.codecogs.com/svg.latex?h(v)=A.v" />
, then the null-space is again the set of all vectors that are sent to the zero vector by <img  src="https://latex.codecogs.com/svg.latex?h"  alt="https://latex.codecogs.com/svg.latex?h" />. Think of this as the set of vectors that lose their identity as h is applied to them.
Note that the null-space is equivalently the set of solutions to the homogeneous equation <img  src="https://latex.codecogs.com/svg.latex?A.v=0"  alt="https://latex.codecogs.com/svg.latex?A.v=0" />


Writing  <img  src="https://latex.codecogs.com/svg.latex?h(v)%20=%20A%20\cdot%20v"  alt="https://latex.codecogs.com/svg.latex?h(v) = A \cdot v" /> , then the null-space is the set of all vectors that are sent to the zero (lose their identity) as <img  src="https://latex.codecogs.com/svg.latex?h"  alt="https://latex.codecogs.com/svg.latex?h" /> is applied to them.


## Most Common Way to Find the Null Space
1. Row Reduction (Gaussian Elimination)
2. Singular Value Decomposition (SVD)
3. Eigenvalue Decomposition: For square matrices, the null space is closely related to the eigenvectors corresponding to the eigenvalue of zero. <img  src="https://latex.codecogs.com/svg.latex?A%20%3D%20U%5CSigma%20V%5E*"  alt="A = U\Sigma V^*" />. The columns of <img  src="https://latex.codecogs.com/svg.latex?V"  alt="V" />  corresponding to zero singular values in <img  src="https://latex.codecogs.com/svg.latex?\Sigma"  alt="\Sigma" />  form an orthonormal basis for the null space of <img  src="https://latex.codecogs.com/svg.latex?A"  alt="A" />


4. QR Decomposition

## 12.1. Example of Calculating Null Space

Example 1:

<br/>

Lets say we have the following matrix:

<img src="https://latex.codecogs.com/svg.image?A=\begin{bmatrix}1%20&%201%20&%202%20&%201%20\\3%20&%201%20&%20%204&%20%204\\4%20&%20-4%20&%200%20&%20%208\\\end{bmatrix}%20\in%20\mathbb{R}^4" alt="https://latex.codecogs.com/svg.image?A=\begin{bmatrix} 1 & 1 & 2 & 1 \\3 & 1 &  4&  4\\ 4 & -4 & 0 &  8\\ \end{bmatrix} \in \mathbb{R}^4">




<img src="https://latex.codecogs.com/svg.image?Ker(A)=\left\{%20x\in%20\mathbb{R}^4%20|%20Ax=0%20\right\}"  alt="https://latex.codecogs.com/svg.image?Ker(A)=\left\{ x\in \mathbb{R}^4 | Ax=0 \right\}" />

By performing row operations we will get the row echelon form:

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}%201%20&%201%20&%202%20&%201%20\\0%20&%20-2%20&%20%20-2&%20%201\\%200%20&%200%20&%200%20&%20%200\\%20\end{bmatrix}" alt="https://latex.codecogs.com/svg.image?\begin{bmatrix} 1 & 1 & 2 & 1 \\0 & -2 &  -2&  1\\ 0 & 0 & 0 &  0\\ \end{bmatrix}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\begin{cases}-2x_2%20-2x_3+x_4=0\\x_2=-\alpha+%20\frac{1}{2}\beta\end{cases}" alt="https://latex.codecogs.com/svg.image?\begin{cases} -2x_2 -2x_3+x_4=0\\ x_2=-\alpha+ \frac{1}{2}\beta \end{cases}" />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\begin{cases}x_1%20-\alpha%20+\frac{1}{2}\beta%20+2\alpha%20+\beta=0%20\\x_1=-\alpha%20-%20\frac{3}{2}\beta%20\\x_2=-\alpha%20+%20\frac{1}{2}\beta\end{cases}" alt="https://latex.codecogs.com/svg.image?\begin{cases}x_1 -\alpha +\frac{1}{2}\beta +2\alpha +\beta=0 \\x_1=-\alpha - \frac{3}{2}\beta \\x_2=-\alpha + \frac{1}{2}\beta\end{cases}" />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\begin{pmatrix}-\alpha%20-%20\frac{3}{2}\beta%20%20\\-\alpha%20+%20\frac{1}{2}\beta%20%20\\\alpha%20\\\beta%20\\\end{pmatrix}=\alpha\begin{pmatrix}-1%20%20\\-1%20%20\\1%20\\0%20\\\end{pmatrix}+\beta\begin{pmatrix}-\frac{3}{2}%20%20\\\frac{1}{2}\\0%20\\1%20\\\end{pmatrix}\alpha,%20\beta%20\in%20\mathbb{R}" alt="https://latex.codecogs.com/svg.image?\begin{pmatrix} -\alpha - \frac{3}{2}\beta  \\ -\alpha + \frac{1}{2}\beta  \\ \alpha \\ \beta \\ \end{pmatrix}=\alpha\begin{pmatrix} -1  \\-1  \\1 \\0 \\\end{pmatrix}+\beta\begin{pmatrix}-\frac{3}{2}  \\\frac{1}{2}\\0 \\1 \\\end{pmatrix}\alpha, \beta \in \mathbb{R}" />

These two column are basis of our kernel.

With Eigen, you can get a basis of the null space using `Eigen::FullPivLU::kernel()` method:

```cpp
Eigen::MatrixXd A(3,4);
A<<1 ,1 ,2, 1 ,
    3,1,4,4,
    4,-4,0,8;


Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
Eigen::MatrixXd A_null_space = lu.kernel();
```
Since `FullPivLU` is [expensive](http://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html), a better alternate is 
to use `CompleteOrthogonalDecomposition.`

```cpp
CompleteOrthogonalDecomposition<Matrix<double, Dynamic, Dynamic> > cod;
cod.compute(A);
std::cout << "rank : " << cod.rank() << "\n";
// Find URV^T
MatrixXd V = cod.matrixZ().transpose();
MatrixXd Null_space = V.block(0, cod.rank(),V.rows(), V.cols() - cod.rank());
MatrixXd P = cod.colsPermutation();
Null_space = P * Null_space; // Unpermute the columns
// The Null space:
std::cout << "The null space: \n" << Null_space << "\n" ;
// Check that it is the null-space:
std::cout << "A * Null_space = \n" << A * Null_space  << '\n';
```

<br/>
Example 2
<br/>

Refs: [1](http://immersivemath.com/ila/ch08_rank/ch08.html#sec_rank_null_space)

# 13. Nullity
The dimension of the kernel of A is called the **nullity** of A



The kernel of L is the vector space of all elements v of V such that 
<img  src="https://latex.codecogs.com/svg.latex?L(v)%20=%200"  alt="https://latex.codecogs.com/svg.latex?L(v) = 0" /> , where 0 denotes the zero vector in W.


We can represent the linear map as matrix multiplication <img  src="https://latex.codecogs.com/svg.latex?A_{m\times%20n}"  alt="https://latex.codecogs.com/svg.latex?A_{m\times n}" /> . The kernel of the linear map L is the set of solutions   <img  src="https://latex.codecogs.com/svg.latex?A\mathbf%20x=\mathbf%200"  alt="https://latex.codecogs.com/svg.latex?A\mathbf x=\mathbf 0" />. 


<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20A\mathbf%20{x}%20=\mathbf%20{0}%20\;\;\Leftrightarrow%20\;\;{\begin{alignedat}{}a_{11}x_{1}&&\;+\;&&a_{12}x_{2}&&\;+\;\cdots%20\;+\;&&a_{1n}x_{n}&&\;=\;&&&0\\a_{21}x_{1}&&\;+\;&&a_{22}x_{2}&&\;+\;\cdots%20\;+\;&&a_{2n}x_{n}&&\;=\;&&&0\\&&&&&&&&&&\vdots%20\%20\;&&&\\a_{m1}x_{1}&&\;+\;&&a_{m2}x_{2}&&\;+\;\cdots%20\;+\;&&a_{mn}x_{n}&&\;=\;&&&0{\text{.}}\\\end{alignedat}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle A\mathbf {x} =\mathbf {0} \;\;\Leftrightarrow \;\;{\begin{alignedat}{}a_{11}x_{1}&&\;+\;&&a_{12}x_{2}&&\;+\;\cdots \;+\;&&a_{1n}x_{n}&&\;=\;&&&0\\a_{21}x_{1}&&\;+\;&&a_{22}x_{2}&&\;+\;\cdots \;+\;&&a_{2n}x_{n}&&\;=\;&&&0\\&&&&&&&&&&\vdots \ \;&&&\\a_{m1}x_{1}&&\;+\;&&a_{m2}x_{2}&&\;+\;\cdots \;+\;&&a_{mn}x_{n}&&\;=\;&&&0{\text{.}}\\\end{alignedat}}}" />

This means to find the kernel of A is we need to solve the above homogeneous equations.


Refs: [1](https://math.unm.edu/~loring/links/linear_s06/nullity.pdf)

# 14. Rank-nullity Theorem

Let <img  src="https://latex.codecogs.com/svg.latex?T\colon%20V\to%20W"  alt="https://latex.codecogs.com/svg.latex?T\colon V\to W" />  be a linear transformation. Then

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\operatorname%20{Rank}%20(T)+\operatorname%20{Nullity}%20(T)=\dim%20V,}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \operatorname {Rank} (T)+\operatorname {Nullity} (T)=\dim V,}" />
Nullity is the complement to the rank of a matrix. 





# 15. The Determinant of The Matrix
Then the determinant of A is the product of the elements of the diagonal of B:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\det(A)={\frac%20{\prod%20\operatorname%20{diag}%20(B)}{d}}.}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \det(A)={\frac {\prod \operatorname {diag} (B)}{d}}.}" />

# 15.1 Interpretation of matrix determinant

The determinant of a matrix has several important interpretations and implications, especially in the context of linear algebra and its applications across various fields such as mathematics, physics, and engineering. Here are the key interpretations:

1. **Geometric Interpretation**:
   - The determinant of a matrix represents the scaling factor of the transformation defined by the matrix. In two dimensions, it tells you how much the area of a shape will change after it's transformed by the matrix. In three dimensions, it tells you how much the volume of a solid will change.
   - When the determinant is **zero**, it means that the transformation squashes the shape into a lower-dimensional space, effectively reducing its volume (or area in 2D) to zero. This implies that the matrix is singular, meaning it does not have an inverse, and the vectors that the matrix is transforming are linearly dependent.
   - A **large determinant** (whether positive or negative) indicates that the shape is scaled up by a large factor in terms of area or volume. A positive determinant means the orientation is preserved (e.g., right-handed to right-handed coordinate system), while a negative determinant indicates a change in orientation (e.g., right-handed to left-handed coordinate system).

2. **Algebraic Interpretation**:
   - The determinant provides a criterion for the invertibility of a matrix. A non-zero determinant means the matrix is invertible (non-singular), while a zero determinant means it is not invertible (singular).
   - The value of the determinant can also be seen in the context of solving systems of linear equations. A zero determinant implies that the system of equations does not have a unique solution, either having no solution or infinitely many solutions.

Let's go through some numerical examples to illustrate these points:

### Example 1: Zero Determinant
Consider the matrix


<img  src="https://latex.codecogs.com/svg.latex?A%20%3D%20%5Cbegin%7Bpmatrix%7D%201%20%26%202%20%5C%5C%202%20%26%204%20%5Cend%7Bpmatrix%7D"  alt=" A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}" />





This matrix represents a linear transformation that maps 2D vectors in a way that makes them linearly dependent. The determinant of \(A\) would be \(0\), indicating that it collapses the plane into a line or point, depending on the vectors it's transforming.

### Example 2: Non-Zero Determinant
Consider the matrix

<img  src="https://latex.codecogs.com/svg.latex?B%20%3D%20%5Cbegin%7Bpmatrix%7D%203%20%26%200%20%5C%5C%200%20%26%202%20%5Cend%7Bpmatrix%7D"  alt=" B = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix} " />



The determinant of <img  src="https://latex.codecogs.com/svg.latex?B"  alt="B" /> would be \(6\), indicating that this transformation scales areas by a factor of <img  src="https://latex.codecogs.com/svg.latex?6"  alt="6" />. It is invertible, and the transformation preserves the orientation of shapes.

Let's calculate these determinants to illustrate the points made.

As calculated:

- The determinant of matrix <img  src="https://latex.codecogs.com/svg.latex?A"  alt="A" /> is <img  src="https://latex.codecogs.com/svg.latex?0"  alt="0" />, which confirms our interpretation that it cannot invert the transformation it represents. This means the matrix maps all input vectors into a lower-dimensional space (in this case, a line or a point), indicating that the vectors are linearly dependent.

- The determinant of matrix <img  src="https://latex.codecogs.com/svg.latex?B"  alt="B" /> is <img  src="https://latex.codecogs.com/svg.latex?6"  alt="6" />, indicating that the transformation it represents scales areas by a factor of <img  src="https://latex.codecogs.com/svg.latex?6"  alt="6" />. This matrix is invertible, and the transformation it represents preserves the orientation of shapes.

Refs: [1](https://www.youtube.com/watch?v=Ip3X9LOh2dk)



# 16. Finding The Inverse of The Matrix
First, add the n × n identity matrix is augmented to the right of A such that we get the following

<img  src="https://latex.codecogs.com/svg.latex?[A%20|%20I]_{n\times%202n}"  alt="https://latex.codecogs.com/svg.latex?[A | I]_{n\times 2n}" /> Now during the elementary row operations, apply the same operations on the identity matrix on the right hand side. At the end teh matrix n the right hand side is the inverse of A.


# 17. The Fundamental Theorem of Linear Algebra



# 18. Permutation Matrix

A permutation matrix is a square binary matrix that has exactly one entry of `1` in each row and each column and `0`s elsewhere.

<img src="https://latex.codecogs.com/svg.image?{\begin{bmatrix}1&0&0&0&0\\0&0&0&1&0\\0&1&0&0&0\\0&0&0&0&1\\0&0&1&0&0\end{bmatrix}}." 
alt="{\begin{bmatrix}1&0&0&0&0\\0&0&0&1&0\\0&1&0&0&0\\0&0&0&0&1\\0&0&1&0&0\end{bmatrix}}" />

# 19. Augmented Matrix


[<< Previous ](4_Advanced_Eigen_Operations.md)  [Home](README.md)  [ Next >>](6_Sparse_Matrices.md)
