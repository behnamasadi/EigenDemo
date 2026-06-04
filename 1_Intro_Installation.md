# Chapter 1 Introduction and Installation
- [About Eigen](#about-eigen)
- [Installation](#installation)
  * [Debian / Ubuntu](#debian--ubuntu)
  * [conda](#conda)
  * [vcpkg](#vcpkg)
  * [Building from source](#building-from-source)
- [Adding Eigen to Your Project](#adding-eigen-to-your-project)
- [Your First Eigen Program](#your-first-eigen-program)


# About Eigen

[Eigen](https://eigen.tuxfamily.org) is a high-level, **header-only** C++
library for linear algebra: matrices, vectors, numerical solvers, and the
algorithms that operate on them. Because it is header-only, there is nothing to
build or link against — you add the headers to your include path, `#include`
them, and Eigen's code is compiled directly into your program.

Eigen is implemented with a technique called **expression templates**. Instead
of evaluating each operation eagerly, Eigen builds an *expression tree* at
compile time and generates custom code that evaluates the whole expression in a
single pass, only when the result is actually needed. This lets it avoid the
temporary objects a naive implementation would create and enables optimizations
such as loop fusion and loop unrolling.

For example, given `w = x + y + z`, Eigen does not materialize a temporary for
`x + y` and another for `(x + y) + z`. It generates a single loop equivalent to
`for (i) w[i] = x[i] + y[i] + z[i]`.

# Installation

Eigen 3.4 or newer is recommended. Pick whichever method matches your platform.

## Debian / Ubuntu

```bash
sudo apt-get install libeigen3-dev
```

This installs the headers under `/usr/include/eigen3` and a CMake config file,
so `find_package(Eigen3)` works out of the box.

## conda

```bash
conda install -c conda-forge eigen
```

## vcpkg

```bash
vcpkg install eigen3
```

## Building from source

Eigen is header-only, so "building" really just means installing the headers
(and the CMake config files). Clone the repository:

```bash
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/usr"
cmake --build build --target install
```

Setting `-DCMAKE_INSTALL_PREFIX="$HOME/usr"` installs Eigen under
`~/usr` instead of `/usr`, so **no root privileges are needed**.

# Adding Eigen to Your Project

In your `CMakeLists.txt`, locate Eigen and link the imported target
`Eigen3::Eigen` to your executable:

```cmake
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE Eigen3::Eigen)
```

Linking `Eigen3::Eigen` automatically adds Eigen's include directories to the
target — you do **not** need a separate `include_directories(...)` call. This is
the modern, target-based approach and is exactly how the examples in this
repository are built (see the project [`CMakeLists.txt`](CMakeLists.txt)).

If you installed Eigen to a custom prefix (the "from source" path above),
point CMake at it when configuring:

```bash
cmake -B build -DCMAKE_PREFIX_PATH="$HOME/usr"
```

# Your First Eigen Program

The full source is in [`src/hello_eigen.cpp`](src/hello_eigen.cpp):

```cpp
#include <Eigen/Dense>
#include <iostream>

int main() {
  // Eigen exposes its version through preprocessor macros.
  std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "."
            << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\n\n";

  Eigen::Matrix2d a;        // a 2x2 matrix of doubles
  a << 1, 2,
       3, 4;                // filled with the comma initializer

  Eigen::Vector2d b(5, 6);  // a 2D column vector

  std::cout << "matrix a:\n" << a << "\n\n";
  std::cout << "vector b:\n" << b << "\n\n";
  std::cout << "a * b:\n" << a * b << std::endl;

  return 0;
}
```

Expected output (the version reflects your installed Eigen):

```
Eigen version: 3.4.0

matrix a:
1 2
3 4

vector b:
5
6

a * b:
17
39
```

If you see this, your installation and CMake setup are working correctly.

[Home](README.md) [Next >>](2_Matrix_Array_Vector_Class.md)
