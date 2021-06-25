- [Newton's Method In Optimization](#newton-s-method-in-optimization)
- [Gauss-Newton Algorithm](#gauss-newton-algorithm)
    + [Example of Gauss-Newton, Inverse Kinematic Problem](#example-of-gauss-newton--inverse-kinematic-problem)
- [Curve Fitting](#curve-fitting)
- [Non Linear Least Squares](#non-linear-least-squares)
- [Non Linear Regression](#non-linear-regression)
- [Levenberg Marquardt](#levenberg-marquardt)

<!--
<img  src=""  alt="https://latex.codecogs.com/svg.latex?" />
-->






# Newton's Method In Optimization
Newton's method is an iterative method for finding the roots of a differentiable function 

<img  src="https://latex.codecogs.com/svg.latex?F"  alt="https://latex.codecogs.com/svg.latex?F" />. Applying Newton's method  to the derivative <img  src="https://latex.codecogs.com/svg.latex?f^\prime"  alt="https://latex.codecogs.com/svg.latex?f^\prime" /> of a twice-differentiable function <img  src="https://latex.codecogs.com/svg.latex?f"  alt="https://latex.codecogs.com/svg.latex?f" /> to find the roots of the derivative, solutions to <img  src="https://latex.codecogs.com/svg.latex?f^\prime(x)=0"  alt="https://latex.codecogs.com/svg.latex?f^\prime(x)=0" />, will give us **critical points** (minima, maxima, or saddle points) 

The second-order Taylor expansion of <img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20f:\mathbb%20{R}%20\to%20\mathbb%20{R}%20}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle f:\mathbb {R} \to \mathbb {R} }" /> around <img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20x_{k}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle x_{k}}" /> is:



<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20f(x_{k}+t)\approx%20f(x_{k})+f%27(x_{k})t+{\frac%20{1}{2}}f%27%27(x_{k})t^{2}.}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle f(x_{k}+t)\approx f(x_{k})+f'(x_{k})t+{\frac {1}{2}}f''(x_{k})t^{2}.}" />


setting the derivative to zero:


<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\displaystyle%200={\frac%20{\rm%20{d}}{{\rm%20{d}}t}}\left(f(x_{k})+f%27(x_{k})t+{\frac%20{1}{2}}f%27%27(x_{k})t^{2}\right)=f%27(x_{k})+f%27%27(x_{k})t,}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \displaystyle 0={\frac {\rm {d}}{{\rm {d}}t}}\left(f(x_{k})+f'(x_{k})t+{\frac {1}{2}}f''(x_{k})t^{2}\right)=f'(x_{k})+f''(x_{k})t,}" />






<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20x_{k+1}=x_{k}-{\frac%20{f%27(x_{k})}{f%27%27(x_{k})}}.}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle x_{k+1}=x_{k}-{\frac {f'(x_{k})}{f''(x_{k})}}.}" />





# Gauss-Newton Algorithm
The Gauss-Newton algorithm is a modification of Newton's method for finding a minimum of a function.
Unlike Newton's method, second derivatives, (which can be challenging to compute), are not required.


Starting with an initial guess <img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\boldsymbol%20{\beta%20}}^{(0)}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\boldsymbol {\beta }}^{(0)}}" />  for the minimum, the method proceeds by the iterations

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\boldsymbol%20{\beta%20}}^{(s+1)}={\boldsymbol%20{\beta%20}}^{(s)}-\left(\mathbf%20{J_{f}}%20^{\mathsf%20{T}}\mathbf%20{J_{f}}%20\right)^{-1}\mathbf%20{J_{f}}%20^{\mathsf%20{T}}\mathbf%20{f}%20\left({\boldsymbol%20{\beta%20}}^{(s)}\right)}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\boldsymbol {\beta }}^{(s+1)}={\boldsymbol {\beta }}^{(s)}-\left(\mathbf {J_{f}} ^{\mathsf {T}}\mathbf {J_{f}} \right)^{-1}\mathbf {J_{f}} ^{\mathsf {T}}\mathbf {f} \left({\boldsymbol {\beta }}^{(s)}\right)}" />


The <img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\left(\mathbf%20{J_{f}}%20^{\mathsf%20{T}}\mathbf%20{J_{f}}%20\right)^{-1}\mathbf%20{J_{f}}%20^{\mathsf%20{T}}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \left(\mathbf {J_{f}} ^{\mathsf {T}}\mathbf {J_{f}} \right)^{-1}\mathbf {J_{f}} ^{\mathsf {T}}}" /> is called Pseudoinverse. To compute this matrix we use the following property:

Let <img  src="https://latex.codecogs.com/svg.latex?A"  alt="https://latex.codecogs.com/svg.latex?A" />  be a 
<img  src="https://latex.codecogs.com/svg.latex?m\times%20n"  alt="https://latex.codecogs.com/svg.latex?m\times n" />  matrix with the SVD:

<img  src="https://latex.codecogs.com/svg.latex?A%20=%20U%20\Sigma%20V^T"  alt="https://latex.codecogs.com/svg.latex?A = U \Sigma V^T" />

and 

<img  src="https://latex.codecogs.com/svg.latex?A^+%20=%20(A^T%20A)^{-1}%20A^T"  alt="https://latex.codecogs.com/svg.latex?A^+ = (A^T A)^{-1} A^T" />


Then, we can write the pseudoinverse as:

<img  src="https://latex.codecogs.com/svg.latex?A^+%20=%20V%20\Sigma^{-1}%20U^T"  alt="https://latex.codecogs.com/svg.latex?A^+ = V \Sigma^{-1} U^T" />


Proof: 

<img  src="https://latex.codecogs.com/svg.latex?\begin{align}%20A^+%20&=%20(A^TA)^{-1}A^T%20\\%20&=(V\Sigma%20U^TU\Sigma%20V^T)^{-1}%20V\Sigma%20U^T%20\\%20&=(V\Sigma^2%20V^T)^{-1}%20V\Sigma%20U^T%20\\%20&=(V^T)^{-1}%20\Sigma^{-2}%20V^{-1}%20V\Sigma%20U^T%20\\%20&=%20V%20\Sigma^{-2}\Sigma%20U^T%20\\%20&=%20V\Sigma^{-1}U^T%20\end{align}"  alt="https://latex.codecogs.com/svg.latex?\begin{align}
A^+ &= (A^TA)^{-1}A^T\\ 
    &=(V\Sigma U^TU\Sigma V^T)^{-1} V\Sigma U^T \\
    &=(V\Sigma^2 V^T)^{-1} V\Sigma U^T \\
    &=(V^T)^{-1} \Sigma^{-2} V^{-1} V\Sigma U^T \\
    &= V \Sigma^{-2}\Sigma U^T \\
    &= V\Sigma^{-1}U^T\\
\end{align}" />


Refs: [1](https://math.stackexchange.com/questions/19948/pseudoinverse-matrix-and-svd)





### Example of Gauss-Newton, Inverse Kinematic Problem
In the following we solve the inverse kinematic of 3 link planner robot.

```
vector_t q=q_start;
vector_t delta_q(3);


double epsilon=1e-3;

int i=0;
double gamma;
double stepSize=10;

while( (distanceError(goal,forward_kinematics(q)).squaredNorm()>epsilon)  && (i<200)  )
{
    Eigen::MatrixXd jacobian=numericalDifferentiationFK(q);
    Eigen::MatrixXd j_pinv=pseudoInverse(jacobian);
    Eigen::VectorXd delta_p=transformationMatrixToPose(goal)-transformationMatrixToPose(forward_kinematics(q) );

    delta_q=j_pinv*delta_p;
    q=q+delta_q;
    i++;
}
```    

Full source code [here](src/3_link_planner_robot.cpp)




# Curve Fitting
You have a function with unknown parameters and a set of sample data (possibly contaminated with noise) from 
that function and you are interested to find the unknown parameters such that the residual (difference between output of your function and sample data) became minimum.

# Non Linear Least Squares
# Non Linear Regression
# Levenberg Marquardt
 The Levenberg-Marquardt algorithm aka the damped least-squares (DLS) method, is used to solve non-linear least squares problems. The LMA is used in many mainly for solving curve-fitting problems. 
The LMA finds only a local minimum  (which may not be the global minimum). The LMA interpolates between the Gauss-Newton algorithm and the method of gradient descent. The LMA is more robust than the GNA, which means that in many cases it finds a solution even if it starts very far off the final minimum. For well-behaved functions and reasonable starting parameters, the LMA tends to be slower than the GNA. LMA can also be viewed as Gauss–Newton using a trust region approach.
