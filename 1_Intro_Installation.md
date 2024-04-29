# Chapter 1 Introduction and Installation
- [About Eigen](#about-eigen)
- [Instillation](#instillation)
- [Adding Eigen to Your Project](#adding-eigen-to-your-project)


# About Eigen
Eigen is a pure header library meaning everything is in the `.hpp` files. Eigen is implemented using the expression templates technique, which builds structures at compile time, where expressions are evaluated only as needed. This means it builds expression trees at compile time and generates custom code to evaluate them
This technique enables the programmer to bypass the normal order of expression evaluation in the C++ and achieve optimizations such as loop fusion and loop unrolling.

# Instillation
Installation is fairly straightforward, on debian based OS:
```
sudo apt-get install libeigen3-dev
```
or if you want to build it from source code just clone it:
```
git clone https://gitlab.com/libeigen/eigen.git
```
make the the build directory:
```
cd eigen
mkdir build 
cd build
```
Now build and install it,  
```
cmake -DCMAKE_CXX_FLAGS=-std=c++1z -DCMAKE_BUILD_TYPE=Release  -DCMAKE_INSTALL_PREFIX:PATH=~/usr .. && make -j8 all install 
```
please note that I have set:
```
-DCMAKE_INSTALL_PREFIX:PATH=~/usr
```
This way the Eigen will be installed `in home/<user-name>/usr` and not in the `/usr` so no root privilege is needed.


# Adding Eigen to Your Project

Now in your CMake file you have to set the `Eigen3_DIR`:


```
set(Eigen3_DIR "$ENV{HOME}/usr/share/eigen3/cmake")

find_package (Eigen3 REQUIRED NO_MODULE)

MESSAGE("EIGEN3_FOUND: " ${EIGEN3_FOUND})
MESSAGE("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
MESSAGE("EIGEN3_VERSION: " ${EIGEN3_VERSION})
MESSAGE("EIGEN3_VERSION_STRING: " ${EIGEN3_VERSION_STRING})


add_executable (example example.cpp)
target_link_libraries (example Eigen3::Eigen)
```

[Home](README.md) [Next >>](2_Matrix_Array_Vector_Class.md)


