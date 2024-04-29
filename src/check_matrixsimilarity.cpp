#include <iostream>
#include <Eigen/Dense>



void checkMatrixsimilarity()
{
    // EXPECT_NEAR should be used element wise
    // This could be also used
    // ASSERT_TRUE(((translation - expectedTranslation).norm() < precision);

    // Pointwise() matcher could be also used
    // EXPECT_THAT(result_array, Pointwise(NearWithPrecision(0.1), expected_array));
    
    
    Eigen::VectorXd pose;

	Eigen::VectorXd expectedPose(3);
	expectedPose<<1.0,0.0,0.0;
    //ASSERT_TRUE(pose.isApprox(expectedPose));
}

int main()
{

}


