# Introduction to Linear Equation

In many applications we have a system of equations
<!--
```math
SE = \frac{\sigma}{\sqrt{n}}
```
-->
<img src="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}a_{11}x_{1}+a_{12}x_{2}+\cdots +a_{1n}x_{n}&=b_{1}\\a_{21}x_{1}+a_{22}x_{2}+\cdots +a_{2n}x_{n}&=b_{2}\\&\ \ \vdots \\a_{m1}x_{1}+a_{m2}x_{2}+\cdots +a_{mn}x_{n}&=b_{m}\end{aligned}}}" /> 


<br>
<br>

Which can be written as a single matrix equation:

<img src="https://latex.codecogs.com/svg.latex? 
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
<img src="https://latex.codecogs.com/svg.latex?A_{m\times n}x_{n\times 1}=b_{m \times 1}" /> 
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

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}a_{11}x_{1}+a_{12}x_{2}+\cdots +a_{1n}x_{n}&=0\\a_{21}x_{1}+a_{22}x_{2}+\cdots +a_{2n}x_{n}&=0\\&\ \ \vdots \\a_{m1}x_{1}+a_{m2}x_{2}+\cdots +a_{mn}x_{n}&=0\end{aligned}}}" /> 

# Solving Linear Equation

Depending on what your matrices looks like, you can choose between various decompositions, and depending on whether you favor speed or accuracy.

# 1) QR Decomposition
## 1-1) Square Matrix
If <img src="https://latex.codecogs.com/svg.latex?A" />  is a real square matrix, then it may be decomposed as:

<img src="https://latex.codecogs.com/svg.latex?A=QR" /> 

<br>
<br>


Where where <img src="https://latex.codecogs.com/svg.latex?  Q" /> is an orthogonal matrix, 
meaning: 
<img src="https://latex.codecogs.com/svg.latex?  Q^{T}=Q^{-1} " />
 and <img src="https://latex.codecogs.com/svg.latex? R" /> is an upper triangular matrix.
Furthermore, if <img src="https://latex.codecogs.com/svg.latex?A" /> is invertible, then the factorization is unique if we require the diagonal elements of <img src="https://latex.codecogs.com/svg.latex?R" />  to be positive.


For complex square matrices, <img src="https://latex.codecogs.com/svg.latex?Q" />  is a unitary matrix, meaning 
<img src="https://latex.codecogs.com/svg.latex?  Q^{*}=Q^{-1} " />

## 1-2) Rectangular Matrix
If <img src="https://latex.codecogs.com/svg.latex?A_{m\times n}" /> where <img src="https://latex.codecogs.com/svg.latex? m \geq  n" /> we can factor it into <img src="https://latex.codecogs.com/svg.latex?m\times m" /> unitary matrix <img src="https://latex.codecogs.com/svg.latex?Q" /> and  an <img src="https://latex.codecogs.com/svg.latex?m\times n" /> upper triangular matrix <img src="https://latex.codecogs.com/svg.latex?R" />. Since after <img src="https://latex.codecogs.com/svg.latex?\left (m-n \right )_{th}" /> row, in <img src="https://latex.codecogs.com/svg.latex?R" /> all elements are entirely zeroes, we can rewrite our equation in the following form:

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle A_{m\times n}=Q _{m\times m}  R_{m\times n} =Q{\begin{bmatrix}R_{1}\\0\end{bmatrix}}={\begin{bmatrix}Q_{1}&Q_{2}\end{bmatrix}}{\begin{bmatrix}R_{1}\\0\end{bmatrix}}=Q_{1}R_{1},}" />


where 
<img src="https://latex.codecogs.com/svg.latex?R_1" /> is an <img src="https://latex.codecogs.com/svg.latex?n\times n" /> upper triangular matrix and <img src="https://latex.codecogs.com/svg.latex?Q_1" /> is <img src="https://latex.codecogs.com/svg.latex? m \times n" />  with orthogonal columns


## 1-3) QL, RQ and LQ Decompositions
We can define <img src="https://latex.codecogs.com/svg.latex?QL" />, <img src="https://latex.codecogs.com/svg.latex?RQ" />, and <img src="https://latex.codecogs.com/svg.latex?LQ" /> decompositions, with <img src="https://latex.codecogs.com/svg.latex?L" /> being a lower triangular matrix.

## 1-4) Computing the QR Decomposition
### 1-4-1) Gram Schmidt Orthogonalization 
Gram–Schmidt process is a method for orthonormalizing a set of vectors. In this process you make every column perpendicular to it's previous columns. Lets first define the **projection operator** by

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle \mathrm {proj} _{\mathbf {u} }(\mathbf {v} )={\frac {\langle \mathbf {u} ,\mathbf {v} \rangle }{\langle \mathbf {u} ,\mathbf {u} \rangle }}{\mathbf {u} }}" />

where <img src="https://latex.codecogs.com/svg.latex?\langle \mathbf{u}, \mathbf{v}\rangle"> denotes the inner product.

Now lets imagine we have the following vectors,  

<img src="images/gram_schmidt1.png">


The Gram–Schmidt process has the followings steps:

<img src="images/gram_schmidt2.png">

<br>
<br>

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}\mathbf {u} _{1}&=\mathbf {v} _{1},&\mathbf {e} _{1}&={\frac {\mathbf {u} _{1}}{\|\mathbf {u} _{1}\|}}\end{aligned}}}">

<br>
<br>

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned} \\\mathbf {u} _{2}&=\mathbf {v} _{2}-\mathrm {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{2}),&\mathbf {e} _{2}&={\frac {\mathbf {u} _{2}}{\|\mathbf {u} _{2}\|}} \end{aligned}}}">
<br>
<br>

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle \mathbf {u} _{k}=\mathbf {v} _{k}-\operatorname {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{k})-\operatorname {proj} _{\mathbf {u} _{2}}(\mathbf {v} _{k})-\cdots -\operatorname {proj} _{\mathbf {u} _{k-1}}(\mathbf {v} _{k})}"/>

<br>
<br>


<img src="images/gram_schmidt3.png">
<br>
<br>



<img src="images/gram_schmidt4.png">
<br>
<br>




<img src="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned} 
\\\mathbf {u} _{3}&=\mathbf {v} _{3}-\mathrm {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{3})-\mathrm {proj} _{\mathbf {u} _{2}}(\mathbf {v} _{3}),&\mathbf {e} _{3}&={\frac {\mathbf {u} _{3}}{\|\mathbf {u} _{3}\|}}
 \end{aligned}}}">


Due to rounding errors, the vectors <img src="https://latex.codecogs.com/svg.latex?\mathbf {u}_{k}"/> are often not quite orthogonal, therefore, it is said that the (classical) Gram–Schmidt process is numerically unstable. This can be stabilized by a small modification, where 
Instead of computing the vector <img src="https://latex.codecogs.com/svg.latex?\mathbf {u}_{k}"/> as

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle \mathbf {u} _{k}=\mathbf {v} _{k}-\operatorname {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{k})-\operatorname {proj} _{\mathbf {u} _{2}}(\mathbf {v} _{k})-\cdots -\operatorname {proj} _{\mathbf {u} _{k-1}}(\mathbf {v} _{k})}"/>



<img src="https://latex.codecogs.com/svg.latex?
{\displaystyle {\begin{aligned}\mathbf {u} _{k}^{(1)}&=\mathbf {v} _{k}-\operatorname {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{k}),\\\mathbf {u} _{k}^{(2)}&=\mathbf {u} _{k}^{(1)}-\operatorname {proj} _{\mathbf {u} _{2}}\left(\mathbf {u} _{k}^{(1)}\right),\\&\;\;\vdots \\\mathbf {u} _{k}^{(k-2)}&=\mathbf {u} _{k}^{(k-3)}-\operatorname {proj} _{\mathbf {u} _{k-2}}\left(\mathbf {u} _{k}^{(k-3)}\right),\\\mathbf {u} _{k}^{(k-1)}&=\mathbf {u} _{k}^{(k-2)}-\operatorname {proj} _{\mathbf {u} _{k-1}}\left(\mathbf {u} _{k}^{(k-2)}\right),\\\mathbf {u} _{k}&={\frac {\mathbf {u} _{k}^{(k-1)}}{\left\|\mathbf {u} _{k}^{(k-1)}\right\|}}\end{aligned}}}" 
/>




<img src="images/gram_schmidt5.png" />
<br>
<br>
<img src="images/gram_schmidt6.png" />
<br>
<br>
<img src="images/gram_schmidt7.png" />
<br>
<br>
<img src="images/gram_schmidt8.png" />
<br>
<br>











<img src="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}
\\\mathbf {u} _{4}&=\mathbf {v} _{4}-\mathrm {proj} _{\mathbf {u} _{1}}(\mathbf {v} _{4})-\mathrm {proj} _{\mathbf {u} _{2}}(\mathbf {v} _{4})-\mathrm {proj} _{\mathbf {u} _{3}}(\mathbf {v} _{4}),&\mathbf {e} _{4}&={\mathbf {u} _{4} \over \|\mathbf {u} _{4}\|}\\&{}\ \ \vdots &&{}\ \ \vdots
 \end{aligned}}}">

 

<img src="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned} 
\\\mathbf {u} _{k}&=\mathbf {v} _{k}-\sum _{j=1}^{k-1}\mathrm {proj} _{\mathbf {u} _{j}}(\mathbf {v} _{k}),&\mathbf {e} _{k}&={\frac {\mathbf {u} _{k}}{\|\mathbf {u} _{k}\|}}.
\end{aligned}}}">




















### 1-4-2) Householder Transformations
# 2) Cholesky Decomposition
# 3) LDU Decomposition
# 4) SVD Decomposition

