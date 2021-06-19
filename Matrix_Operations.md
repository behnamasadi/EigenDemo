- [Matrix Arithmetic](#matrix-arithmetic)
  * [Addition/Subtraction](#addition-subtraction)
  * [Scalar Multiplication/ Division](#scalar-multiplication--division)
  * [Transposition and Conjugation](#transposition-and-conjugation)
  * [Multiplication, Dot And Cross Product](#multiplication--dot-and-cross-product)
- [Coefficient-Wise Operations](#coefficient-wise-operations)
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
  * [Adding/ Subtracting A Constant](#adding--subtracting-a-constant)
  * [Masking Elements](#masking-elements)
  
# Matrix Arithmetic
## Addition/Subtraction

## Scalar Multiplication/ Division

## Transposition and Conjugation

## Multiplication, Dot And Cross Product


# Coefficient-Wise Operations

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
## Adding/ Subtracting A Constant

```
matrix.array() - 2;
```

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
