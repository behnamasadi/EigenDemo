#  Chapter 4 Advanced Eigen Operations
- [Memory Alignment](#memory-alignment)
- [Passing Eigen objects by value to functions](#passing-eigen-objects-by-value-to-functions)
- [Aliasing](#aliasing)
- [Memory Mapping](#memory-mapping)
  * [Eigen matrix from std::vector](#eigen-matrix-from-stdvector)
- [Unary Expression](#unary-expression)
- [Eigen Functor](#eigen-functor)

The runnable code for this chapter is in
[`src/memory_mapping.cpp`](src/memory_mapping.cpp),
[`src/unaryExpr.cpp`](src/unaryExpr.cpp) and
[`src/eigen_functor.cpp`](src/eigen_functor.cpp).

# Memory Alignment

To use SIMD instructions (SSE, AVX, NEON), Eigen needs some fixed-size objects
to be aligned in memory (to 16 or 32 bytes). This is handled automatically for
local variables, but two situations need care:

1. **A fixed-size Eigen member inside your own class.** A class containing, for
   example, an `Eigen::Vector4d` or `Eigen::Matrix4f` member may need an
   overloaded aligned `operator new`. Add the macro:
   ```cpp
   class Foo {
     Eigen::Vector4d v;
   public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   };
   ```
2. **A `std::vector` (or other STL container) of fixed-size Eigen types.** Use
   Eigen's aligned allocator:
   ```cpp
   std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> v;
   ```

These caveats apply only to *fixed-size, vectorizable* types (sizes that are a
multiple of 16 bytes, e.g. `Vector4f`, `Vector2d`, `Matrix4d`). Dynamic-size
types (`MatrixXd`, `VectorXd`) and small types like `Vector3f` are unaffected.
Building with C++17 (which this project does) also lets the compiler use aligned
`new`, removing many of these requirements.

# Passing Eigen objects by value to functions

**Never pass fixed-size vectorizable Eigen objects by value.** Passing by value
forces a copy and can break the alignment guarantees above. Pass by `const`
reference instead:

```cpp
// Bad: copies, and may misalign a Vector4d.
void foo(Eigen::Vector4d v);

// Good: no copy.
void foo(const Eigen::Vector4d &v);
```

To write a function that accepts **any** matrix expression (a matrix, a block, a
sub-vector, the result of an operation) without forcing evaluation or copies,
take a template parameter deriving from `Eigen::MatrixBase`, or an
`Eigen::Ref`:

```cpp
// Accepts any expression; no temporary is created.
template <typename Derived>
void printSize(const Eigen::MatrixBase<Derived> &m) {
  std::cout << m.rows() << "x" << m.cols() << "\n";
}

// Ref binds to a matrix or a compatible block without copying.
double sumOf(const Eigen::Ref<const Eigen::MatrixXd> &m) { return m.sum(); }
```

Ref: [Writing Functions Taking Eigen Types as Parameters](https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html)

# Aliasing

Aliasing happens when the same matrix appears on both sides of an assignment and
the entries are read and written in an overlapping way. The classic trap is
transpose:

```cpp
Eigen::MatrixXi m(2, 2);
m << 1, 2,
     3, 4;

m = m.transpose();          // WRONG: aliasing corrupts the result
m.transposeInPlace();       // correct
// or force a temporary:
m = m.transpose().eval();   // correct
```

The same applies to `m = m.adjoint()`, `v = v.reverse()`, etc. — use the
`*InPlace()` variant or `.eval()`.

Matrix **multiplication** is the exception: Eigen assumes aliasing for the
matrix product and introduces a temporary automatically, so `m = m * m` is safe.
If you know there is no aliasing and want to skip the temporary, use
`noalias()`:

```cpp
c.noalias() = a * b; // promises that c does not alias a or b
```

# Memory Mapping
In many applications your data already lives in some other data structure and
you need to run linear-algebra operations on it. Suppose you store points as:
```cpp
struct point {
    double a;
    double b;
};
```
and a shape is a `std::vector<point>` you want to transform with an affine
matrix. One (costly) approach is to iterate over the container and copy each
value into an Eigen matrix. A better approach is to **map** the existing memory
into an Eigen matrix without copying, using `Eigen::Map`.

The mapped buffer must outlive the `Map`, and you must not invalidate it (no
`clear()`/`resize()` that reallocates) while the `Map` is in use.

## Eigen matrix from std::vector

```cpp
std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
```

`Map` reads the buffer in column-major order by default, or row-major if you
ask for it:

```cpp
Eigen::Map<Eigen::Matrix<float, 3, 3>> einMatColMajor(data.data());
Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> einMatRowMajor(data.data());
```

The same memory therefore yields two different matrices:

```
column-major view:        row-major view:
1 4 7                     1 2 3
2 5 8                     4 5 6
3 6 9                     7 8 9
```

A `Map` is writable too: assigning to `einMatColMajor(i, j)` modifies the
underlying `std::vector`. See `eigenMapExample()` in the source for mapping
`std::vector<Eigen::Vector3d>` and arrays of plain structs.

# Unary Expression

`unaryExpr` applies a unary function to every coefficient and returns a new
expression. The function can be a lambda or a function pointer. It is a `const`
operation — it does not modify in place, so assign the result back to update:

```cpp
Eigen::ArrayXd x = Eigen::ArrayXd::Random(5);

// Lambda: map each coefficient to 0/1 depending on its sign.
x = x.unaryExpr([](double e) { return e < 0.0 ? 0.0 : 1.0; });

// Function pointer (std::ptr_fun was removed in C++17 — pass the function):
double ramp(double v) { return v > 0 ? v : 0; }
x.unaryExpr(std::ref(ramp));
```

# Eigen Functor

A *functor* is an object that behaves like a function — it carries state and
exposes a call/evaluation method. Functors are the mechanism Eigen's solvers use
to receive a user-defined cost function: the `unsupported`
`NonLinearOptimization` and `NumericalDiff` modules (used in Chapter 9) expect
you to provide a struct with an `operator()` and the relevant compile-time
sizes.

Here is the plain C++ idea — a struct holding its inputs and exposing an
evaluation method `f()`:

```cpp
template <typename scalar>
struct product_functor {
  product_functor(scalar a, scalar b) : m_a(a), m_b(b) {}
  scalar f() const { return m_a * m_b; }
private:
  scalar m_a, m_b;
};

template <typename functor_type>
void call_and_print_return_value(const functor_type &functor_object) {
  std::cout << "The result is: " << functor_object.f() << std::endl;
}

call_and_print_return_value(product_functor<float>(0.2f, 0.4f)); // 0.08
```

In Chapter 9 you will see the Eigen-specific shape of a functor (deriving from a
base that declares `InputsAtCompileTime`/`ValuesAtCompileTime` and implementing
`operator()(const InputType&, ValueType&)`) used to drive Levenberg–Marquardt.

[<< Previous ](3_Matrix_Operations.md)  [Home](README.md)  [ Next >>](5_Dense_Linear_Problems_And_Decompositions.md)
