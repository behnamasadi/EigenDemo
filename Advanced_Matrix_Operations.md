- [Matrix Broadcasting](#matrix-broadcasting)
- [Matrix Condition](#matrix-condition)
- [Check Matrix Similarity](#check-matrix-similarity)
- [Matrix Rank](#matrix-rank)
- [Memory Mapping](#memory-mapping)
- [Unary Expression](#unary-expression)
- [Eigen Functor](#eigen-functor)


# Matrix Broadcasting
# Matrix Condition
# Check Matrix Similarity 
# Matrix Rank
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

