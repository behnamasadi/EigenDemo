#  Chapter 9 Numerical Optimization
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
Newton's method is an iterative method for finding the roots of a differentiable function <img  src="https://latex.codecogs.com/svg.latex?F"  alt="https://latex.codecogs.com/svg.latex?F" />. Applying Newton's method  to the derivative <img  src="https://latex.codecogs.com/svg.latex?f^\prime"  alt="https://latex.codecogs.com/svg.latex?f^\prime" /> of a twice-differentiable function <img  src="https://latex.codecogs.com/svg.latex?f"  alt="https://latex.codecogs.com/svg.latex?f" /> to find the roots of the derivative, (solutions to <img  src="https://latex.codecogs.com/svg.latex?f^\prime(x)=0"  alt="https://latex.codecogs.com/svg.latex?f^\prime(x)=0" />) will give us **critical points** (minima, maxima, or saddle points) 

The second-order Taylor expansion of <img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20f:\mathbb%20{R}%20\to%20\mathbb%20{R}%20}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle f:\mathbb {R} \to \mathbb {R} }" /> around <img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20x_{k}}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle x_{k}}" /> is:



<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20f(x_{k}+t)\approx%20f(x_{k})+f%27(x_{k})t+{\frac%20{1}{2}}f%27%27(x_{k})t^{2}.}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle f(x_{k}+t)\approx f(x_{k})+f'(x_{k})t+{\frac {1}{2}}f''(x_{k})t^{2}.}" />


setting the derivative to zero:


<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20\displaystyle%200={\frac%20{\rm%20{d}}{{\rm%20{d}}t}}\left(f(x_{k})+f%27(x_{k})t+{\frac%20{1}{2}}f%27%27(x_{k})t^{2}\right)=f%27(x_{k})+f%27%27(x_{k})t,}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle \displaystyle 0={\frac {\rm {d}}{{\rm {d}}t}}\left(f(x_{k})+f'(x_{k})t+{\frac {1}{2}}f''(x_{k})t^{2}\right)=f'(x_{k})+f''(x_{k})t,}" />

This will give us:


<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20t=-{\frac%20{f%27(x_{k})}{f%27%27(x_{k})}}.}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle t=-{\frac {f'(x_{k})}{f''(x_{k})}}.}" />

We start the next point <img  src="https://latex.codecogs.com/svg.latex?x_{k+1}"  alt="https://latex.codecogs.com/svg.latex?x_{k+1}" /> at where the second order approximation is zero, so
<img  src="https://latex.codecogs.com/svg.latex?x_{k+1}=t+x_{k}"  alt="https://latex.codecogs.com/svg.latex?x_{k+1}=t+x_{k}" />

By putting everything together:


<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20x_{k+1}=x_{k}-{\frac%20{f%27(x_{k})}{f%27%27(x_{k})}}.}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle x_{k+1}=x_{k}-{\frac {f'(x_{k})}{f''(x_{k})}}.}" />

So we iterate this until the changes are small enough.

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


# Quasi-Newton Method

# Curve Fitting
You have a function <img  src="https://latex.codecogs.com/svg.latex?y%20=%20f(x,%20\boldsymbol%20\beta)"  alt="https://latex.codecogs.com/svg.latex?y = f(x, \boldsymbol \beta)" /> with <img  src="https://latex.codecogs.com/svg.latex?n"  alt="https://latex.codecogs.com/svg.latex?n" /> unknown parameters, <img  src="https://latex.codecogs.com/svg.latex?\boldsymbol%20\beta=(\beta_1,\beta_2,...\beta_n)"  alt="https://latex.codecogs.com/svg.latex?\boldsymbol \beta=(\beta_1,\beta_2,...\beta_n)" />  and a set of sample data (possibly contaminated with noise) from 
that function and you are interested to find the unknown parameters such that the residual (difference between output of your function and sample data) <img  src="https://latex.codecogs.com/svg.latex?\boldsymbol%20r=(r_1,r_2,...r_m)"  alt="https://latex.codecogs.com/svg.latex?\boldsymbol r=(r_1,r_2,...r_m)" />
 became minimum. We construct a new function, where it is sum square of all residuals:
 
<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20S({\boldsymbol%20{\beta%20}})=\sum%20_{i=1}^{m}r_{i}({\boldsymbol%20{\beta%20}})^{2}.}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle S({\boldsymbol {\beta }})=\sum _{i=1}^{m}r_{i}({\boldsymbol {\beta }})^{2}.}" />



residuals are:




<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20r_{i}({\boldsymbol%20{\beta%20}})=y_{i}-f\left(x_{i},{\boldsymbol%20{\beta%20}}\right).}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle r_{i}({\boldsymbol {\beta }})=y_{i}-f\left(x_{i},{\boldsymbol {\beta }}\right).}" />



We start with an initial guess for <img  src="https://latex.codecogs.com/svg.latex?\boldsymbol%20{\beta%20}^0"  alt="https://latex.codecogs.com/svg.latex?\boldsymbol {\beta }^0" />

Then we proceeds by the iterations:

<img  src="https://latex.codecogs.com/svg.latex?{\displaystyle%20{\boldsymbol%20{\beta%20}}^{(s+1)}={\boldsymbol%20{\beta%20}}^{(s)}-\left(\mathbf%20{J_{r}}%20^{\mathsf%20{T}}\mathbf%20{J_{r}}%20\right)^{-1}\mathbf%20{J_{r}}%20^{\mathsf%20{T}}\mathbf%20{r}%20\left({\boldsymbol%20{\beta%20}}^{(s)}\right)}"  alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\boldsymbol {\beta }}^{(s+1)}={\boldsymbol {\beta }}^{(s)}-\left(\mathbf {J_{r}} ^{\mathsf {T}}\mathbf {J_{r}} \right)^{-1}\mathbf {J_{r}} ^{\mathsf {T}}\mathbf {r} \left({\boldsymbol {\beta }}^{(s)}\right)}" />



## Example of Substrate Concentration Cuver Fitting

Lets say we have the following dataset:


|i	|1	|2	|3	|4	|5	|6	|7
|---    |---    |---    |---    |---    |---    |---    |---|
|[S]	|0.038	|0.194	|0.425	|0.626	|1.253	|2.500	|3.740
|Rate	|0.050	|0.127	|0.094	|0.2122	|0.2729	|0.2665	|0.3317

And we have the folliwng function to fit the data:

<img  src="https://latex.codecogs.com/svg.latex?y=\frac{\beta_1%20\times%20x}{\beta_2+x}"  alt="https://latex.codecogs.com/svg.latex?y=\frac{\beta_1 \times x}{\beta_2+x}" />


So our <img  src="https://latex.codecogs.com/svg.latex?\mathbf%20r_{2\times7}"  alt="https://latex.codecogs.com/svg.latex?\mathbf r_{2\times7}" /> is:


<img  src="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix}%20r_1=y_1%20-%20\frac{\beta_1%20\times%20x_1}{\beta_2+x_1}\\%20r_2=y_2%20-%20\frac{\beta_1%20\times%20x_2}{\beta_2+x_2}\\%20r_3=y_3%20-%20\frac{\beta_1%20\times%20x_3}{\beta_2+x_3}\\%20r_4=y_4%20-%20\frac{\beta_1%20\times%20x_4}{\beta_2+x_4}\\%20r_5=y_5%20-%20\frac{\beta_1%20\times%20x_5}{\beta_2+x_5}\\%20r_6=y_6%20-%20\frac{\beta_1%20\times%20x_6}{\beta_2+x_6}\\%20r_7=y_7%20-%20\frac{\beta_1%20\times%20x_7}{\beta_2+x_7}\\%20\end{matrix}\right."  alt="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix}
r_1=y_1 - \frac{\beta_1 \times x_1}{\beta_2+x_1}\\ 
r_2=y_2 - \frac{\beta_1 \times x_2}{\beta_2+x_2}\\ 
r_3=y_3 - \frac{\beta_1 \times x_3}{\beta_2+x_3}\\ 
r_4=y_4 - \frac{\beta_1 \times x_4}{\beta_2+x_4}\\ 
r_5=y_5 - \frac{\beta_1 \times x_5}{\beta_2+x_5}\\ 
r_6=y_6 - \frac{\beta_1 \times x_6}{\beta_2+x_6}\\ 
r_7=y_7 - \frac{\beta_1 \times x_7}{\beta_2+x_7}\\ 
\end{matrix}\right." />


and the jacobian is <img  src="https://latex.codecogs.com/svg.latex?\mathbf%20J_{7\times2}"  alt="https://latex.codecogs.com/svg.latex?\mathbf J_{7\times2}" />:


<img  src="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix}%20\frac{\sigma%20r_i}{\sigma%20%20\beta_1}=%20\frac{-x_i}{\beta_2+x_i}%20\\%20\frac{\sigma%20r_i}{\sigma%20\beta_2}%20=%20\frac{\beta_1*x_i}{(\beta_2%20\times%20x_i)^2}%20\end{matrix}\right."  alt="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix}
\frac{\sigma r_i}{\sigma  \beta_1}= \frac{-x_i}{\beta_2+x_i}  
\\ 
\frac{\sigma r_i}{\sigma \beta_2} = \frac{\beta_1*x_i}{(\beta_2 \times x_i)^2}  
\end{matrix}\right." />:


Now let's implement it with Eigen, First a functor that calculate the 
<img  src="https://latex.codecogs.com/svg.latex?\mathbf%20r"  alt="https://latex.codecogs.com/svg.latex?\mathbf r" /> given <img  src="https://latex.codecogs.com/svg.latex?\boldsymbol%20{\beta%20}^0"  alt="https://latex.codecogs.com/svg.latex?\boldsymbol {\beta }" />:
```
struct SubstrateConcentrationFunctor : Functor<double>
{
    SubstrateConcentrationFunctor(Eigen::MatrixXd points): Functor<double>(points.cols(),points.rows())
    {
        this->Points = points;
    }

    int operator()(const Eigen::VectorXd &z, Eigen::VectorXd &r) const
    {
        double x_i,y_i,beta1,beta2;
        for(unsigned int i = 0; i < this->Points.rows(); ++i)
        {
            y_i=this->Points.row(i)(1);
            x_i=this->Points.row(i)(0);
            beta1=z(0);
            beta2=z(1);
            r(i) =y_i-(beta1*x_i) /(beta2+x_i);
        }

        return 0;
    }
    Eigen::MatrixXd Points;

    int inputs() const { return 2; } // There are two parameters of the model, beta1, beta2
    int values() const { return this->Points.rows(); } // The number of observations
};
```

Now we have to set teh data:
```
//the last column in the matrix should be "y"
Eigen::MatrixXd points(7,2);

points.row(0)<< 0.038,0.050;
points.row(1)<<0.194,0.127;
points.row(2)<<0.425,0.094;
points.row(3)<<0.626,0.2122;
points.row(4)<<1.253,0.2729;
points.row(5)<<2.500,0.2665;
points.row(6)<<3.740,0.3317;
```
Let's create an instance of our functor and compute the derivatves numerically:
```
SubstrateConcentrationFunctor functor(points);
Eigen::NumericalDiff<SubstrateConcentrationFunctor,Eigen::NumericalDiffMode::Central> numDiff(functor);
```
Set the initial values for beta:
```
Eigen::VectorXd beta(2);
beta<<0.9,0.2;
```
And the main loop:
```
    //βᵏ⁺¹=βᵏ- (JᵀJ)⁻¹Jᵀr(βᵏ)
    for(int i=0;i<10;i++)
    {
        numDiff.df(beta,J);
        std::cout<<"J: \n" << J<<std::endl;
        functor(beta,r);
        std::cout<<"r: \n" << r<<std::endl;
        beta=beta-(J.transpose()*J).inverse()*J.transpose()*r ;
    }
    std::cout<<"beta: \n" << beta<<std::endl;
```


# Non Linear Least Squares
# Non Linear Regression
# Levenberg Marquardt
 The Levenberg-Marquardt algorithm aka the damped least-squares (DLS) method, is used to solve non-linear least squares problems. The LMA is used in many mainly for solving curve-fitting problems. 
The LMA finds only a local minimum  (which may not be the global minimum). The LMA interpolates between the Gauss-Newton algorithm and the method of gradient descent. The LMA is more robust than the GNA, which means that in many cases it finds a solution even if it starts very far off the final minimum. For well-behaved functions and reasonable starting parameters, the LMA tends to be slower than the GNA. LMA can also be viewed as Gauss–Newton using a trust region approach.



[<< Previous ](8_Differentiation.md)   [Home](README.md)  

