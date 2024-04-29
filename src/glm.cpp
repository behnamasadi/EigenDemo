#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/scalar_constants.hpp> // glm::pi
#include <glm/mat4x4.hpp>               // glm::mat4
#include <glm/vec3.hpp>                 // glm::vec3
#include <glm/vec4.hpp>                 // glm::vec4

glm::mat4 camera(float Translate, glm::vec2 const &Rotate) {
  glm::mat4 Projection =
      glm::perspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, 0.1f, 100.f);
  glm::mat4 View =
      glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -Translate));
  View = glm::rotate(View, Rotate.y, glm::vec3(-1.0f, 0.0f, 0.0f));
  View = glm::rotate(View, Rotate.x, glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 Model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
  return Projection * View * Model;
}

int main() {
/*
eigen to glm
EigenToGlmMat
https://stackoverflow.com/questions/63429179/eigen-and-glm-products-produce-different-results

https://gist.github.com/podgorskiy/04a3cb36a27159e296599183215a71b0


Quaternion to Matrix using glm
https://stackoverflow.com/questions/38145042/quaternion-to-matrix-using-glm


glm::toQuat(
glm::to_string(GLR).c_str()
glm::toMat3


*/



    float Translate=1.0;
    glm::vec2  Rotate;

    glm::mat4 v=camera( Translate, Rotate);
}
