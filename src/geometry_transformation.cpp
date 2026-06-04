#include <iostream>
#include <Eigen/Dense>

//Transform, Translation, Scaling, Rotation2D and 3D rotations (Quaternion, AngleAxis)

#if KDL_FOUND==1
#include <kdl/frames.hpp>
#endif

//The value of KDL_FOUND has been set via target_compile_definitions in CMake




void transformExample()
{
/*
The class Eigen::Transform  represents either
1) an affine, or
2) a projective transformation
using homogenous calculus.

For instance, an affine transformation A is composed of a linear part L and a translation t such that transforming a point p by A is equivalent to:

p' = L * p + t

Using homogeneous vectors:

    [p'] = [L t] * [p] = A * [p]
    [1 ]   [0 1]   [1]       [1]

Ref: https://stackoverflow.com/questions/35416880/what-does-transformlinear-return-in-the-eigen-library

Difference Between Projective and Affine Transformations
1) The projective transformation does not preserve parallelism, length, and angle. But it still preserves collinearity and incidence.
2) Since the affine transformation is a special case of the projective transformation,
it has the same properties. However unlike projective transformation, it preserves parallelism.

Ref: https://www.graphicsmill.com/docs/gm5/Transformations.htm
*/

    float arrVertices [] = { -1.0 , -1.0 , -1.0 ,
    1.0 , -1.0 , -1.0 ,
    1.0 , 1.0 , -1.0 ,
    -1.0 , 1.0 , -1.0 ,
    -1.0 , -1.0 , 1.0 ,
    1.0 , -1.0 , 1.0 ,
    1.0 , 1.0 , 1.0 ,
    -1.0 , 1.0 , 1.0};
    Eigen::MatrixXf mVertices = Eigen::Map < Eigen::Matrix <float , 3 , 8 > > ( arrVertices ) ;
    Eigen::Transform <float , 3 , Eigen::Affine > t = Eigen::Transform <float , 3 , Eigen::Affine >::Identity();
    t.scale ( 0.8f ) ;
    t.rotate ( Eigen::AngleAxisf (0.25f * M_PI , Eigen::Vector3f::UnitX () ) ) ;
    t.translate ( Eigen::Vector3f (1.5 , 10.2 , -5.1) ) ;
    std::cout << t * mVertices.colwise().homogeneous () << std::endl;
}


// Build an affine matrix by post-multiplying (rotate after translate). This
// applies the rotations in the local/rotating frame ("Euler angle" style).
Eigen::Matrix4f createAffinematrix(float a, float b, float c,
                                   Eigen::Vector3f trans) {
  Eigen::Transform<float, 3, Eigen::Affine> t;
  t = Eigen::Translation<float, 3>(trans);
  t.rotate(Eigen::AngleAxis<float>(a, Eigen::Vector3f::UnitX()));
  t.rotate(Eigen::AngleAxis<float>(b, Eigen::Vector3f::UnitY()));
  t.rotate(Eigen::AngleAxis<float>(c, Eigen::Vector3f::UnitZ()));
  return t.matrix();
}

// The same rotations applied by pre-multiplying instead. The difference between
// the two is like the difference between "fixed angle" and "Euler angle"
// conventions. See https://www.youtube.com/watch?v=09xVHo1JudY
Eigen::Matrix4f createAffinematrixPrerotate(float a, float b, float c,
                                            Eigen::Vector3f trans) {
  Eigen::Transform<float, 3, Eigen::Affine> t;
  t = Eigen::AngleAxis<float>(c, Eigen::Vector3f::UnitZ());
  t.prerotate(Eigen::AngleAxis<float>(b, Eigen::Vector3f::UnitY()));
  t.prerotate(Eigen::AngleAxis<float>(a, Eigen::Vector3f::UnitX()));
  t.pretranslate(trans);
  return t.matrix();
}

int main() {
  std::cout << "Transforming the 8 cube vertices (scale, rotate, translate):\n";
  transformExample();

  Eigen::Vector3f trans(1.5f, 10.2f, -5.1f);
  std::cout << "\ncreateAffinematrix (post-rotate):\n"
            << createAffinematrix(0.1f, 0.2f, 0.3f, trans) << "\n";
  std::cout << "\ncreateAffinematrixPrerotate (pre-rotate):\n"
            << createAffinematrixPrerotate(0.1f, 0.2f, 0.3f, trans) << "\n";
  return 0;
}
