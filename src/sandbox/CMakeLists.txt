add_executable(lm src/lm.cpp)
target_link_libraries (lm ${EIGEN3_LIBRARIES}  )

add_executable(lm1 src/lm1.cpp)
target_link_libraries (lm1 ${EIGEN3_LIBRARIES}  )

add_executable(lm2 src/lm2.cpp)
target_link_libraries (lm ${EIGEN3_LIBRARIES}  )

add_executable(numerical_diff src/numerical_diff.cpp)
target_link_libraries (numerical_diff ${EIGEN3_LIBRARIES}  )
