#include <iostream>
#include <Eigen/Dense>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

struct LMFunctor
{
    // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
    int operator()(const Eigen::VectorXf &model_parameter, Eigen::VectorXf &fvec) const
    {
        // 'model_parameter' has dimensions n x 1
        // It contains the current estimates for the parameters.

        // 'fvec' has dimensions m x 1
        // It will contain the error for each data point.

        float aParam = model_parameter(0);
        float bParam = model_parameter(1);
        float cParam = model_parameter(2);

        for (int i = 0; i < values(); i++)
        {
            float xValue = measuredValues(i, 0);
            float yValue = measuredValues(i, 1);

            fvec(i) = yValue - (aParam * xValue * xValue + bParam * xValue + cParam);
        }

        return 0;
    }

    // Compute the jacobian of the errors, numerically
    int df(const Eigen::VectorXf &model_parameter, Eigen::MatrixXf &fjac) const
    {
        // 'model_parameter' has dimensions n x 1
        // It contains the current estimates for the parameters.

        // 'fjac' has dimensions m x n
        // It will contain the jacobian of the errors, calculated numerically in this case.

        float epsilon;
        epsilon = 1e-5f;

        for (int i = 0; i < model_parameter.size(); i++)
        {
            Eigen::VectorXf xPlus(model_parameter);
            xPlus(i) += epsilon;
            Eigen::VectorXf xMinus(model_parameter);
            xMinus(i) -= epsilon;

            Eigen::VectorXf fvecPlus(values());
            operator()(xPlus, fvecPlus);

            Eigen::VectorXf fvecMinus(values());
            operator()(xMinus, fvecMinus);

            Eigen::VectorXf fvecDiff(values());
            fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

            fjac.block(0, i, values(), 1) = fvecDiff;
        }
    }



////     Compute the jacobian of the errors, analytically
//    int df(const Eigen::VectorXf &model_parameter, Eigen::MatrixXf &fjac) const
//    {
//        // 'model_parameter' has dimensions n x 1
//        // It contains the current estimates for the parameters.

//        // 'fjac' has dimensions m x n
//        // It will contain the jacobian of the errors, calculated numerically in this case.

//        //f(x) = ax² + bx + c
//        /*
//         jaccobian mxn =[x_1**2, x_1, 1
//                         x_2**2, x_2, 1

//                         x_m**2, x_m, 1]

//        */

//        for(int i=0;i<m;i++)
//        {
//            fjac(i,0)=-measuredValues(i, 0)*measuredValues(i, 0);
//            fjac(i,1)=-measuredValues(i, 0);
//            fjac(i,2)=-1;

//        }

//    }

    // Number of data points, i.e. values.
    int m;

    // Returns 'm', the number of values.
    int values() const { return m; }

    // The number of parameters, i.e. inputs.
    int n;

    // Returns 'n', the number of inputs.
    int inputs() const { return n; }

    Eigen::MatrixXf measuredValues;

};

void levenbergMarquardt()
{
//https://medium.com/@sarvagya.vaish/levenberg-marquardt-optimization-part-2-5a71f7db27a0
/*

n is the number of parameters in the non-linear system. The API refers to the parameters as inputs.
f(x) = ax² + bx + c has 3 parameters: a, b, c
n = 3;

m is the number of measured (x, f(x)) data points. These are our constrains and the API refers to them as values.
m = 100;

x is an n-by-1 vector containing values for the n parameters. In the beginning this vector is filled with the initial guess for the parameters.
It gets updated during the optimization and in the end it contains the solution.
*/

    int n = 3;
    Eigen::VectorXf model_parameter(n);

    int m = 100;

/*fvec is a m-by-1 vector containing errors for each of the m data points.
An error is defined as the difference between the measured data and the value of the function.
Reminder - LM will minimize the sum of squared errors.

fjac is a m-by-n matrix containing the jacobian of the errors.
The jacobian can be calculated analytically or numerically. More on this later.

*/

    Eigen::MatrixXf measuredValues(m, 2);


    measuredValues<<-10.00,-685.80,
    -9.50,-647.10,
    -9.00,-602.00,
    -8.50,-548.90,
    -8.00,-524.20,
    -7.50,-490.10,
    -7.00,-430.60,
    -6.50,-412.10,
    -6.00,-345.20,
    -5.50,-344.10,
    -5.00,-306.00,
    -4.50,-255.50,
    -4.00,-226.00,
    -3.50,-195.90,
    -3.00,-139.00,
    -2.50,-129.50,
    -2.00,-84.80,
    -1.50,-86.50,
    -1.00,-61.00,
    -0.50,-18.70,
    0.00,-3.40,
    0.50,20.90,
    1.00,41.60,
    1.50,86.30,
    2.00,98.80,
    2.50,122.70,
    3.00,149.00,
    3.50,149.10,
    4.00,158.80,
    4.50,188.10,
    5.00,214.20,
    5.50,223.70,
    6.00,231.40,
    6.50,257.50,
    7.00,269.80,
    7.50,274.50,
    8.00,287.40,
    8.50,278.70,
    9.00,304.40,
    9.50,323.30,
    10.00,316.60,
    10.50,329.10,
    11.00,299.20,
    11.50,333.50,
    12.00,340.60,
    12.50,311.10,
    13.00,313.80,
    13.50,302.50,
    14.00,324.80,
    14.50,316.70,
    15.00,298.20,
    15.50,292.10,
    16.00,286.00,
    16.50,297.30,
    17.00,266.80,
    17.50,269.30,
    18.00,276.20,
    18.50,257.90,
    19.00,221.80,
    19.50,216.50,
    20.00,196.80,
    20.50,211.50,
    21.00,194.20,
    21.50,152.70,
    22.00,148.80,
    22.50,128.10,
    23.00,93.60,
    23.50,83.50,
    24.00,50.40,
    24.50,28.30,
    25.00,20.40,
    25.50,-34.10,
    26.00,-53.80,
    26.50,-80.90,
    27.00,-78.60,
    27.50,-128.10,
    28.00,-164.20,
    28.50,-178.30,
    29.00,-207.40,
    29.50,-241.50,
    30.00,-299.00,
    30.50,-324.10,
    31.00,-348.60,
    31.50,-408.30,
    32.00,-446.80,
    32.50,-486.10,
    33.00,-499.40,
    33.50,-558.90,
    34.00,-582.60,
    34.50,-651.70,
    35.00,-706.20,
    35.50,-727.10,
    36.00,-799.60,
    36.50,-839.50,
    37.00,-879.00,
    37.50,-931.50,
    38.00,-969.80,
    38.50,-1041.50,
    39.00,-1066.40,
    39.50,-1119.30;

    model_parameter(0) = 0.0;     // initial value for ‘a’
    model_parameter(1) = 0.0;     // initial value for ‘b’
    model_parameter(2) = 0.0;     // initial value for ‘c’





    LMFunctor functor;
    functor.m = m;
    functor.n = n;
    functor.measuredValues=measuredValues;

    Eigen::LevenbergMarquardt<LMFunctor, float> lm(functor);
    int ret =lm.minimize(model_parameter);

    std::cout << lm.iter << std::endl;
    std::cout << ret << std::endl;

    std::cout << "Optimization results" << std::endl;
    std::cout << "\ta: " << model_parameter(0) << std::endl;
    std::cout << "\tb: " << model_parameter(1) << std::endl;
    std::cout << "\tc: " << model_parameter(2) << std::endl;
}

int main(int argc, char *argv[])
{
    return 0;
}
