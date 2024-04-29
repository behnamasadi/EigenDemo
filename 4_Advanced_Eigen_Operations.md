#  Chapter 4 Advanced Eigen Operations
- [Memory Alignment](#memory-alignment)
- [Passing Eigen objects by value to functions](#passing-eigen-objects-by-value-to-functions)
- [Aliasing](#aliasing)
- [Memory Mapping](#memory-mapping)
- [Unary Expression](#unary-expression)
- [Eigen Functor](#eigen-functor)

# Memory Alignment
# Passing Eigen objects by value to functions

Refs: [1](https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html)

# Aliasing

# Memory Mapping
In many applications, your data has been stored in different data structures and you need to perform some operations on the data. Suppose you have the following class for presenting your points and you have shape which is `std::vector` of such point and now you need to perform an affine matrix transformation on your shape. 
```
struct point {
    double a;
    double b;
};
```
One costly approach would be to iterate over your container and copy your data to some Eigen matrix. But there is a better approach. Eigen enables you to map the memory (of your existing data) into your matrices without copying.
# Unary Expression
# Eigen Functor

[<< Previous ](3_Matrix_Operations.md)  [Home](README.md)  [ Next >>](5_Dense_Linear_Problems_And_Decompositions.md)
