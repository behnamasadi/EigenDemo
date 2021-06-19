#include <iostream>
#include <Eigen/Dense>


//unaryExpr, Lambda Expression, function pointer ,in place update
double ramp(double x)
{
    if (x > 0)
        return x;
    else
        return 0;
}

void unaryExprExample()
{
    //unaryExpr is a const function so it will not give you the possibility to modify values in place
    Eigen::ArrayXd x = Eigen::ArrayXd::Random(5);
    std::cout<<x <<std::endl;
    x = x.unaryExpr([](double elem) // changed type of parameter
    {
        return elem < 0.0 ? 0.0 : 1.0; // return instead of assignment
    });
    std::cout<<x <<std::endl;
    std::cout << x.unaryExpr(std::ptr_fun(ramp)) << std::endl;

}


int main(int argc, char **argv)
{
    return 0;
}

