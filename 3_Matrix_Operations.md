- [Matrix Arithmetic](#matrix-arithmetic)
  * [Addition/Subtraction Matrices/ Scalar](#addition-subtraction-matrices--scalar)
  * [Scalar Multiplication/ Division](#scalar-multiplication--division)
  * [Multiplication, Dot And Cross Product](#multiplication--dot-and-cross-product)
  * [Transposition and Conjugation](#transposition-and-conjugation)
- [Coefficient-Wise Operations](#coefficient-wise-operations)
  * [Multiplication](#multiplication)
  * [Absolute](#absolute)
  * [Power, Root](#power--root)
  * [Log, Exponential](#log--exponential)
  * [Min, Mix of Two Matrices](#min--mix-of-two-matrices)
  * [Check Matrices Similarity](#check-matrices-similarity)
  * [Finite, Inf, NaN](#finite--inf--nan)
  * [Sinusoidal](#sinusoidal)
  * [Floor, Ceil, Round](#floor--ceil--round)
  * [Masking Elements](#masking-elements)
- [Reductions](#reductions)
  * [Minimum/ Maximum Element In The Matrix](#minimum--maximum-element-in-the-matrix)
  * [Minimum/ Maximum Element Row-wise/Col-wise in the Matrix](#minimum--maximum-element-row-wise-col-wise-in-the-matrix)
  * [Sum Of All Elements](#sum-of-all-elements)
  * [Mean Of The Matrix](#mean-of-the-matrix)
  * [Mean Of The Matrix Row-wise/Col-wise](#mean-of-the-matrix-row-wise-col-wise)
  * [The Trace Of The Matrix](#the-trace-of-the-matrix)
  * [The Multiplication Of All Elements](#the-multiplication-of-all-elements)
  * [Norm 2 of The Matrix](#norm-2-of-the-matrix)
  * [Norm Infinity Of The Matrix](#norm-infinity-of-the-matrix)
  * [Checking If All Elements Are Positive](#checking-if-all-elements-are-positive)
  * [Checking If Any Elements Is Positive](#checking-if-any-elements-is-positive)
  * [Counting Elements](#counting-elements)
  * [Matrix Condition Number and Numerical Stability](#matrix-condition-number)
  * [Matrix Rank](#matrix-rank)
- [Broadcasting](#broadcasting)


# Matrix Arithmetic
## Addition/Subtraction Matrices/ Scalar


```
matrix.array() - 2;
```

## Scalar Multiplication/ Division



## Multiplication, Dot And Cross Product

## Transposition and Conjugation

# Coefficient-Wise Operations
## Multiplication
## Absolute
 .abs()
 
## Power, Root
 sqrt() 
array1.sqrt()                 sqrt(array1)
array1.square()
array1.cube()
array1.pow(array2)            pow(array1,array2)
array1.pow(scalar)            pow(array1,scalar)
                              pow(scalar,array2)


## Log, Exponential

array1.log()                  log(array1)
array1.log10()                log10(array1)
array1.exp()                  exp(array1)

## Min, Mix of Two Matrices
 .min(.) 
 max
 
 If you have two arrays of the same size, you can call .min(.) to construct the array whose coefficients are the minimum of the corresponding coefficients of the two given arrays. 
 
## Check Matrices Similarity 

## Finite, Inf, NaN


array1.isFinite()             isfinite(array1)
array1.isInf()                isinf(array1)
array1.isNaN()   
 
## Sinusoidal

array1.sin()                  sin(array1)
array1.cos()                  cos(array1)
array1.tan()                  tan(array1)
array1.asin()                 asin(array1)
array1.acos()                 acos(array1)
array1.atan()                 atan(array1)
array1.sinh()                 sinh(array1)
array1.cosh()                 cosh(array1)
array1.tanh()                 tanh(array1)
array1.arg()                  arg(array1)

## Floor, Ceil, Round 
array1.floor()                floor(array1)
array1.ceil()                 ceil(array1)
array1.round()                round(aray1)



## Masking Elements
In the following, we want to replace the element of our matrix, with an element from the either matrices `P` or `Q`. If the value in our matrix is smaller than a threshold, we replace it with an element from `P` otherwise from `Q`.

```
int cols, rows;
cols=2; rows=3;
Eigen::MatrixXf R=Eigen::MatrixXf::Random(rows, cols);

Eigen::MatrixXf Q=Eigen::MatrixXf::Zero(rows, cols);
Eigen::MatrixXf P=Eigen::MatrixXf::Constant(rows, cols,1.0);

double threshold=0.5;
Eigen::MatrixXf masked=(R.array() < threshold).select(P,Q ); // (R < threshold ? P : Q)
```

# Reductions

## Minimum/ Maximum Element In The Matrix

```
int min_element_row_index,min_element_col_index;
matrix.minCoeff(&min_element_row_index,&min_element_col_index);
```
## Minimum/ Maximum Element Row-wise/Col-wise in the Matrix
```
matrix.rowwise().maxCoeff();
```
## Sum Of All Elements
```
matrix.sum();
```
## Mean Of The Matrix
```
matrix.mean()
```
## Mean Of The Matrix Row-wise/Col-wise 
```
matrix.colwise().mean();
```
 
## The Trace Of The Matrix
```
matrix.trace();
```

## The Multiplication Of All Elements 
```
matrix.prod();
```

## Norm 2 of The Matrix 
```
matrix.lpNorm<2>()
```

## Norm Infinity Of The Matrix
```
matrix.lpNorm<Eigen::Infinity>()
```
## Checking If All Elements Are Positive
```
(matrix.array()>0).all();
```
## Checking If Any Elements Is Positive
```
(matrix.array()>0).any();
```
## Counting Elements
```
(matrix.array()>1).count();
```
## Matrix Condition Number and Numerical Stability

## Matrix Rank
The column rank of a matrix is maximal number of linearly independent columns of that matrix. The column rank of a matrix is in the dimension of the column space, while the row rank of A is the dimension of the row space. It can be proven that <img  src="https://latex.codecogs.com/svg.latex? \text{column rank} =\text{row rank}"  alt="https://latex.codecogs.com/svg.latex? \text{column rank} =\text{row rank}" />

 


Full rank: if its rank equals the largest possible for a matrix of the same dimensions, which is the lesser of the number of rows and columns
 
 A matrix is said to be rank-deficient if it does not have full rank. The rank deficiency of a matrix is the difference between the lesser between the number of rows and columns, and the rank.


Decomposing the matrix is the most common way to get the rank

Gaussian Elimination (row reduction):

This method can also be used to compute the rank of a matrix
the determinant of a square matrix
 inverse of an invertible matrix

Row echelon form: means that Gaussian elimination has operated on the rows
Column echelon form
# Broadcasting
