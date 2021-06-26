- [Introduction to Linear Equation](#introduction-to-linear-equation)
  * [Solution set](#solution-set)
  * [Underdetermined System](#underdetermined-system)
  * [Overdetermined System](#overdetermined-system)
  * [Determined](#determined)
  * [Homogeneous vs Non-homogeneous](#homogeneous-vs-non-homogeneous)
- [Matrices Decompositions](#matrices-decompositions)
  * [QR Decomposition](#qr-decomposition)
    + [Square Matrix](#square-matrix)
    + [Rectangular Matrix](#rectangular-matrix)
    + [Computing the QR Decomposition](#computing-the-qr-decomposition)
      - [Gram Schmidt Orthogonalization](#gram-schmidt-orthogonalization)
      - [Householder Transformations](#householder-transformations)
  * [QL, RQ and LQ Decompositions](#ql--rq-and-lq-decompositions)
  * [Cholesky Decomposition](#cholesky-decomposition)
  * [LDU Decomposition](#ldu-decomposition)
  * [SVD Decomposition](#svd-decomposition)
  * [Eigen Value Eigen Vector](#eigen-value-eigen-vector)
  * [Basis of Null space and Kernel](#basis-of-null-space-and-kernel)
      - [Linear Map](#linear-map)
    + [Vector space](#vector-space)
    + [Null Space](#null-space)
- [Solving Linear Equation](#solving-linear-equation)
  * [Gaussian Elimination](#gaussian-elimination)
    + [Row echelon form](#row-echelon-form)
    + [Trapezoidal Matrix](#trapezoidal-matrix)
    + [Example of The Gaussian Elimination Algorithm](#example-of-the-gaussian-elimination-algorithm)
    + [The Determinant of The Matrix](#the-determinant-of-the-matrix)
    + [Finding The Inverse of The Matrix](#finding-the-inverse-of-the-matrix)
    + [Computing Ranks and Bases](#computing-ranks-and-bases)
  * [Conclusion on Computing Rank](#conclusion-on-computing-rank)

# Introduction to Linear Equation

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

## Solution set 

A linear system may behave in any one of three possible ways:

- The system has infinitely many solutions.
- The system has a single unique solution.
- The system has no solution.


The answer of a linear system is determined by the relationship between the number of equations and the number of unknowns.

## Underdetermined System
a system with fewer equations than unknowns has infinitely many solutions, but it may have no solution. Such a system is known as an underdetermined system.

## Overdetermined System
A system with more equations than unknowns is called as an overdetermined system.

## Determined
A system with the same number of equations and unknowns.



Depending on what your matrices looks like, you can choose between various decompositions, and depending on whether you favor speed or accuracy.

## Homogeneous vs Non-homogeneous 
A system of linear equations is homogeneous if all of the constant terms are zero.

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}a_{11}x_{1}+a_{12}x_{2}+\cdots%20+a_{1n}x_{n}&=0\\a_{21}x_{1}+a_{22}x_{2}+\cdots%20+a_{2n}x_{n}&=0\\&\%20\%20\vdots%20\\a_{m1}x_{1}+a_{m2}x_{2}+\cdots%20+a_{mn}x_{n}&=0\end{aligned}}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}a_{11}x_{1}+a_{12}x_{2}+\cdots +a_{1n}x_{n}&=0\\a_{21}x_{1}+a_{22}x_{2}+\cdots +a_{2n}x_{n}&=0\\&\ \ \vdots \\a_{m1}x_{1}+a_{m2}x_{2}+\cdots +a_{mn}x_{n}&=0\end{aligned}}}" /> 


# Matrices Decompositions
Depending on what your matrices looks like, you can choose between various decompositions, and depending on whether you favor speed or accuracy.

##  QR Decomposition
###  Square Matrix
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

### Rectangular Matrix
If <img src="https://latex.codecogs.com/svg.latex?A_{m\times%20n}" alt="https://latex.codecogs.com/svg.latex?A_{m\times n}" /> where <img src="https://latex.codecogs.com/svg.latex?%20m%20\geq%20%20n" alt="https://latex.codecogs.com/svg.latex? m \geq  n" /> we can factor it into <img src="https://latex.codecogs.com/svg.latex?m\times%20m" alt="https://latex.codecogs.com/svg.latex?m\times m" /> unitary matrix <img src="https://latex.codecogs.com/svg.latex?Q" /> and  an <img src="https://latex.codecogs.com/svg.latex?m\times%20n" alt="https://latex.codecogs.com/svg.latex?m\times n" /> upper triangular matrix <img src="https://latex.codecogs.com/svg.latex?R" />. Since after <img src="https://latex.codecogs.com/svg.latex?\left%20(m-n%20\right%20)_{th}" alt="https://latex.codecogs.com/svg.latex?\left (m-n \right )_{th}" /> row, in <img src="https://latex.codecogs.com/svg.latex?R" /> all elements are entirely zeroes, we can rewrite our equation in the following form:

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20A_{m\times%20n}=Q%20_{m\times%20m}%20%20R_{m\times%20n}%20=Q{\begin{bmatrix}R_{1}\\0\end{bmatrix}}={\begin{bmatrix}Q_{1}&Q_{2}\end{bmatrix}}{\begin{bmatrix}R_{1}\\0\end{bmatrix}}=Q_{1}R_{1},}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle A_{m\times n}=Q _{m\times m}  R_{m\times n} =Q{\begin{bmatrix}R_{1}\\0\end{bmatrix}}={\begin{bmatrix}Q_{1}&Q_{2}\end{bmatrix}}{\begin{bmatrix}R_{1}\\0\end{bmatrix}}=Q_{1}R_{1},}" />


where 
<img src="https://latex.codecogs.com/svg.latex?R_1" /> is an <img src="https://latex.codecogs.com/svg.latex?n\times%20n" alt="https://latex.codecogs.com/svg.latex?n\times n" /> upper triangular matrix and <img src="https://latex.codecogs.com/svg.latex?Q_1" /> is <img src="https://latex.codecogs.com/svg.latex?%20m%20\times%20n" alt="https://latex.codecogs.com/svg.latex? m \times n" />  with orthogonal columns



### Computing the QR Decomposition
#### Gram Schmidt Orthogonalization 
Gram–Schmidt process is a method for orthonormalizing a set of vectors. In this process you make every column perpendicular to it's previous columns. Lets first define the **projection operator** by

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\mathrm%20{proj}%20_{\mathbf%20{u}%20}(\mathbf%20{v}%20)={\frac%20{\langle%20\mathbf%20{u}%20,\mathbf%20{v}%20\rangle%20}{\langle%20\mathbf%20{u}%20,\mathbf%20{u}%20\rangle%20}}{\mathbf%20{u}%20}}" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \mathrm {proj} _{\mathbf {u} }(\mathbf {v} )={\frac {\langle \mathbf {u} ,\mathbf {v} \rangle }{\langle \mathbf {u} ,\mathbf {u} \rangle }}{\mathbf {u} }}" />

where <img src="https://latex.codecogs.com/svg.latex?\langle%20\mathbf{u},%20\mathbf{v}\rangle" alt="https://latex.codecogs.com/svg.latex?\langle \mathbf{u}, \mathbf{v}\rangle"> denotes the inner product.

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



#### Householder Transformations
## QL, RQ and LQ Decompositions
We can define <img src="https://latex.codecogs.com/svg.latex?QL" />, <img src="https://latex.codecogs.com/svg.latex?RQ" />, and <img src="https://latex.codecogs.com/svg.latex?LQ" /> decompositions, with <img src="https://latex.codecogs.com/svg.latex?L" /> being a lower triangular matrix.

## Cholesky Decomposition

Cholesky decomposition is a decomposition of a Hermitian, positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose


## LDU Decomposition
## SVD Decomposition
## Eigen Value Eigen Vector
## Basis of Null space and Kernel


First let review some definitions:
#### Linear Map
Let  <img  src="https://latex.codecogs.com/svg.latex?V"  alt="https://latex.codecogs.com/svg.latex?V" /> and  <img  src="https://latex.codecogs.com/svg.latex?W"  alt="https://latex.codecogs.com/svg.latex?W" /> be vector spaces over the same field  <img  src="https://latex.codecogs.com/svg.latex?K"  alt="https://latex.codecogs.com/svg.latex?K" />. A function <img  src="https://latex.codecogs.com/svg.latex?f:V\to%20W"  alt="https://latex.codecogs.com/svg.latex?f:V\to W" />  is said to be a linear map if for any two vectors
 <img  src="https://latex.codecogs.com/svg.latex?{\textstyle%20\mathbf%20{u}%20,\mathbf%20{v}%20\in%20V}"  alt="https://latex.codecogs.com/svg.latex?{\textstyle \mathbf {u} ,\mathbf {v} \in V}" />  and any scalar 
  <img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20c\in%20K}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle c\in K}" />  the following two conditions are satisfied:

- 1. Additivity

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20f(\mathbf%20{u}%20+\mathbf%20{v}%20)=f(\mathbf%20{u}%20)+f(\mathbf%20{v}%20)}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle f(\mathbf {u} +\mathbf {v} )=f(\mathbf {u} )+f(\mathbf {v} )}" />


- 2. Operation of scalar multiplication

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20f(c\mathbf%20{u}%20)=cf(\mathbf%20{u}%20)}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle f(c\mathbf {u} )=cf(\mathbf {u} )}" />

### Vector space

### Null Space 

The kernel (null space or nullspace) of a linear map, is the linear subspace of the domain of the map which is mapped to the zero vector. Lets assume <img  src="https://latex.codecogs.com/svg.latex?L"  alt="https://latex.codecogs.com/svg.latex?L" /> is a linear map between two vector spaces <img  src="https://latex.codecogs.com/svg.latex?V"  alt="https://latex.codecogs.com/svg.latex?V" />  and  <img  src="https://latex.codecogs.com/svg.latex?W"  alt="https://latex.codecogs.com/svg.latex?W" /> 
<br/>


 <img  src="https://latex.codecogs.com/svg.latex?L%20:%20V%20\to%20W"  alt="https://latex.codecogs.com/svg.latex?L : V \to W" /> 
<br/> 
 
The kernel of L is the vector space of all elements v of V such that 
<img  src="https://latex.codecogs.com/svg.latex?L(v)%20=%200"  alt="https://latex.codecogs.com/svg.latex?L(v) = 0" /> , where 0 denotes the zero vector in W.


We can represent the linear map as matrix multiplication <img  src="https://latex.codecogs.com/svg.latex?A_{m\times%20n}"  alt="https://latex.codecogs.com/svg.latex?A_{m\times n}" /> . The kernel of the linear map L is the set of solutions   <img  src="https://latex.codecogs.com/svg.latex?A\mathbf%20x=\mathbf%200"  alt="https://latex.codecogs.com/svg.latex?A\mathbf x=\mathbf 0" />. 


<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20A\mathbf%20{x}%20=\mathbf%20{0}%20\;\;\Leftrightarrow%20\;\;{\begin{alignedat}{}a_{11}x_{1}&&\;+\;&&a_{12}x_{2}&&\;+\;\cdots%20\;+\;&&a_{1n}x_{n}&&\;=\;&&&0\\a_{21}x_{1}&&\;+\;&&a_{22}x_{2}&&\;+\;\cdots%20\;+\;&&a_{2n}x_{n}&&\;=\;&&&0\\&&&&&&&&&&\vdots%20\%20\;&&&\\a_{m1}x_{1}&&\;+\;&&a_{m2}x_{2}&&\;+\;\cdots%20\;+\;&&a_{mn}x_{n}&&\;=\;&&&0{\text{.}}\\\end{alignedat}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle A\mathbf {x} =\mathbf {0} \;\;\Leftrightarrow \;\;{\begin{alignedat}{}a_{11}x_{1}&&\;+\;&&a_{12}x_{2}&&\;+\;\cdots \;+\;&&a_{1n}x_{n}&&\;=\;&&&0\\a_{21}x_{1}&&\;+\;&&a_{22}x_{2}&&\;+\;\cdots \;+\;&&a_{2n}x_{n}&&\;=\;&&&0\\&&&&&&&&&&\vdots \ \;&&&\\a_{m1}x_{1}&&\;+\;&&a_{m2}x_{2}&&\;+\;\cdots \;+\;&&a_{mn}x_{n}&&\;=\;&&&0{\text{.}}\\\end{alignedat}}}" />

This means to find the kernel of A is we need to solve the above homogeneous equations.


Writing  <img  src="https://latex.codecogs.com/svg.latex?h(v)%20=%20A%20\cdot%20v"  alt="https://latex.codecogs.com/svg.latex?h(v) = A \cdot v" /> , then the null-space is the set of all vectors that are sent to the zero (lose their identity) as <img  src="https://latex.codecogs.com/svg.latex?h"  alt="https://latex.codecogs.com/svg.latex?h" /> is applied to them.



then the null-space is again the set of all vectors that are sent to the zero vector by ℎ. Think of this as the set of vectors that lose their identity as ℎ is applied to them.


### Nullity
The dimension of the kernel of A is called the **nullity** of A


### Rank-nullity Theorem

Let <img  src="https://latex.codecogs.com/svg.latex?T\colon%20V\to%20W"  alt="https://latex.codecogs.com/svg.latex?T\colon V\to W" />  be a linear transformation. Then

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\operatorname%20{Rank}%20(T)+\operatorname%20{Nullity}%20(T)=\dim%20V,}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \operatorname {Rank} (T)+\operatorname {Nullity} (T)=\dim V,}" />
Nullity is the complement to the rank of a matrix. 



# Solving Linear Equation
## Gaussian Elimination
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


### Row echelon form
A matrix is in row echelon form if:

- All rows consisting of only zeroes are at the bottom.
- 2 The leading coefficient (also called the pivot) of a nonzero row is always strictly to the right of the leading coefficient of the row above it.

This matrix is in reduced row echelon form, which shows that the left part of the matrix is not always an identity matrix:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left[{\begin{array}{ccccc}1&0&a_{1}&0&b_{1}\\0&1&a_{2}&0&b_{2}\\0&0&0&1&b_{3}\end{array}}\right]}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left[{\begin{array}{ccccc}1&0&a_{1}&0&b_{1}\\0&1&a_{2}&0&b_{2}\\0&0&0&1&b_{3}\end{array}}\right]}" />






The matrix:

<img  src="https://latex.codecogs.com/svg.latex?\begin{pmatrix}1&4&7\\0&2&3\end{pmatrix}"  alt="https://latex.codecogs.com/svg.latex?\begin{pmatrix}1&2&3\\0&4&5\end{pmatrix}" />

is echelon, but not triangular (because not square). 

The matrix:

<img  src="https://latex.codecogs.com/svg.latex?\begin{pmatrix}1&4&7\\0&0&2\\0&0&4\end{pmatrix}"  alt="https://latex.codecogs.com/svg.latex?\begin{pmatrix}1&2&3\\0&0&4\\0&0&5\end{pmatrix}" />

is triangular, but not echelon (because the leading entry 4 is not to the right of the leading entry 2).
For non-singular square matrices, "row echelon" and "upper triangular" are equivalent.

### Trapezoidal Matrix  
A non-square  matrix with zeros above (below) the diagonal is called a lower (upper) trapezoidal matrix.

### Example of The Gaussian Elimination Algorithm

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

### The Determinant of The Matrix
Then the determinant of A is the product of the elements of the diagonal of B:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\det(A)={\frac%20{\prod%20\operatorname%20{diag}%20(B)}{d}}.}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \det(A)={\frac {\prod \operatorname {diag} (B)}{d}}.}" />

### Finding The Inverse of The Matrix
First, add the n × n identity matrix is augmented to the right of A such that we get the following

<img  src="https://latex.codecogs.com/svg.latex?[A%20|%20I]_{n\times%202n}"  alt="https://latex.codecogs.com/svg.latex?[A | I]_{n\times 2n}" /> Now during the elementary row operations, apply the same operations on the identity matrix on the right hand side. At the end teh matrix n the right hand side is the inverse of A.


### Computing Ranks and Bases
A common approach to find the rank of a matrix is to reduce it row echelon form, and count the number of non-zero elements in main diagonal.

For example, the matrix A given by

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20A={\begin{bmatrix}1&2&1\\-2&-3&1\\3&5&0\end{bmatrix}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle A={\begin{bmatrix}1&2&1\\-2&-3&1\\3&5&0\end{bmatrix}}}" />


following elementary row operations:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\begin{aligned}{\begin{bmatrix}1&2&1\\-2&-3&1\\3&5&0\end{bmatrix}}&\xrightarrow%20{2R_{1}+R_{2}\to%20R_{2}}%20{\begin{bmatrix}1&2&1\\0&1&3\\3&5&0\end{bmatrix}}\xrightarrow%20{-3R_{1}+R_{3}\to%20R_{3}}%20{\begin{bmatrix}1&2&1\\0&1&3\\0&-1&-3\end{bmatrix}}\\&\xrightarrow%20{R_{2}+R_{3}\to%20R_{3}}%20\,\,{\begin{bmatrix}1&2&1\\0&1&3\\0&0&0\end{bmatrix}}\xrightarrow%20{-2R_{2}+R_{1}\to%20R_{1}}%20{\begin{bmatrix}1&0&-5\\0&1&3\\0&0&0\end{bmatrix}}~.\end{aligned}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}{\begin{bmatrix}1&2&1\\-2&-3&1\\3&5&0\end{bmatrix}}&\xrightarrow {2R_{1}+R_{2}\to R_{2}} {\begin{bmatrix}1&2&1\\0&1&3\\3&5&0\end{bmatrix}}\xrightarrow {-3R_{1}+R_{3}\to R_{3}} {\begin{bmatrix}1&2&1\\0&1&3\\0&-1&-3\end{bmatrix}}\\&\xrightarrow {R_{2}+R_{3}\to R_{3}} \,\,{\begin{bmatrix}1&2&1\\0&1&3\\0&0&0\end{bmatrix}}\xrightarrow {-2R_{2}+R_{1}\to R_{1}} {\begin{bmatrix}1&0&-5\\0&1&3\\0&0&0\end{bmatrix}}~.\end{aligned}}}" />


There are two non-zero rows in the final matrix and therefore the rank of matrix is 2.

## Conclusion on Computing Rank
In practice, due to floating point error on computers,  Gaussian elimination (LU decomposition) can be unreliable, therefore rank-revealing decomposition such as RRQR factorization (rank-revealing QR which is QR decomposition with pivoting) should be used. The singular value decomposition (SVD) can be used, but it is not an efficient method to do so.



