#include <Eigen/Dense>
#include <iostream>

// unaryExpr applies a unary function (a lambda or a function pointer) to every
// coefficient and returns a new expression. It is a const operation, so it does
// not modify the array in place; assign the result back if you want to update.
double ramp(double x) { return x > 0 ? x : 0; }

void unaryExprExample() {
  Eigen::ArrayXd x = Eigen::ArrayXd::Random(5);
  std::cout << "x =\n" << x.transpose() << std::endl;

  // With a lambda: turn each coefficient into 0/1 depending on its sign.
  Eigen::ArrayXd step =
      x.unaryExpr([](double elem) { return elem < 0.0 ? 0.0 : 1.0; });
  std::cout << "step(x) =\n" << step.transpose() << std::endl;

  // With a function pointer (std::ptr_fun was removed in C++17; pass the
  // function directly instead).
  std::cout << "ramp(x) =\n"
            << x.unaryExpr(std::ref(ramp)).transpose() << std::endl;
}

int main() {
  unaryExprExample();
  return 0;
}
