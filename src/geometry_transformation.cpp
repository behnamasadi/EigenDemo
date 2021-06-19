#include <iostream>
#include <Eigen/Dense>

//Transform, Translation, Scaling, Rotation2D and 3D rotations (Quaternion, AngleAxis)

#if KDL_FOUND==1
#include <kdl/frames.hpp>
#endif

//The value of KDL_FOUND has been set via target_compile_definitions in CMake


Eigen::Matrix3d eulerAnglesToRotationMatrix(double roll, double pitch,double yaw)
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d rotationMatrix = q.matrix();
    return rotationMatrix;
}

void transformation()
{
/*
Great Tutorial:
http://planning.cs.uiuc.edu/node102.html
http://euclideanspace.com/maths/geometry/rotations/conversions/index.htm

Tait–Bryan angles: Z1Y2X3 in the wiki page:
https://en.wikipedia.org/wiki/Euler_angles
  yaw:
      A yaw is a counterclockwise rotation of alpha about the  z-axis. The
  rotation matrix is given by

      R_z

      |cos(alpha) -sin(alpha) 0|
      |sin(apha)   cos(alpha) 0|
      |    0            0     1|

  pitch:
      R_y
      A pitch is a counterclockwise rotation of  beta about the  y-axis. The
  rotation matrix is given by

      |cos(beta)  0   sin(beta)|
      |0          1       0    |
      |-sin(beta) 0   cos(beta)|

  roll:
      A roll is a counterclockwise rotation of  gamma about the  x-axis. The
  rotation matrix is given by
      R_x
      |1          0           0|
      |0 cos(gamma) -sin(gamma)|
      |0 sin(gamma)  cos(gamma)|



      It is important to note that   R_z R_y R_x performs the roll first, then the pitch, and finally the yaw
      Roration matrix: R_z*R_y*R_x

*/
/////////////////////////////////////Rotation Matrix (Tait–Bryan)///////////////////////////////
    double roll, pitch, yaw;
    roll=M_PI/2;
    pitch=M_PI/2;
    yaw=0;//M_PI/6;
    std::cout << "Roll : " <<  roll << std::endl;
    std::cout << "Pitch : " << pitch  << std::endl;
    std::cout << "Yaw : " << yaw  << std::endl;

/////////////////////////////////////Rotation Matrix (Tait–Bryan)///////////////////////////////

    // Roll, Pitch, Yaw to Rotation Matrix
    //Eigen::AngleAxis<double> rollAngle(roll, Eigen::Vector3d(1,0,0));

    std::cout << "Roll : " <<  roll << std::endl;
    std::cout << "Pitch : " << pitch  << std::endl;
    std::cout << "Yaw : " << yaw  << std::endl;



    // Roll, Pitch, Yaw to Rotation Matrix
    //Eigen::AngleAxis<double> rollAngle(roll, Eigen::Vector3d(1,0,0));
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());



//////////////////////////////////////// Quaternion ///////////////////////////////////////////
    Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;

    
    //Quaternion to Rotation Matrix
    Eigen::Matrix3d rotationMatrix = q.matrix();
    std::cout << "3x3 Rotation Matrix" << std::endl;

    std::cout << rotationMatrix << std::endl;

    Eigen::Quaterniond quaternion_mat(rotationMatrix);
    std::cout << "Quaternion X: " << quaternion_mat.x() << std::endl;
    std::cout << "Quaternion Y: " << quaternion_mat.y() << std::endl;
    std::cout << "Quaternion Z: " << quaternion_mat.z() << std::endl;
    std::cout << "Quaternion W: " << quaternion_mat.w() << std::endl;



//////////////////////////////////////// Rodrigues ///////////////////////////////////////////


    //Rotation Matrix to Rodrigues
    Eigen::AngleAxisd rodrigues(rotationMatrix );
    std::cout<<"Rodrigues Angle:\n"<<rodrigues.angle() <<std::endl;

    std::cout<<"Rodrigues Axis:" <<std::endl;

    std::cout<<rodrigues.axis().x() <<std::endl;
    std::cout<<rodrigues.axis().y() <<std::endl;
    std::cout<<rodrigues.axis().z() <<std::endl;





    Eigen::Vector3d vector3d(2.3,3.1,1.7);
    Eigen::Vector3d vector3dNormalized=vector3d.normalized();
    double theta=M_PI/7;
    Eigen::AngleAxisd angleAxisConversion(theta,vector3dNormalized);
    Eigen::Matrix3d rotationMatrixConversion;

    // Angle Axis (Rodrigues) to Rotation Matrix
    rotationMatrixConversion=angleAxisConversion.toRotationMatrix();

    
    //Rotation Matrix to Quaternion
    
    Eigen::Quaterniond QuaternionConversion(rotationMatrixConversion);

    //Rotation Matrix to Euler Angle (Proper)
    Eigen::Vector3d euler_angles = rotationMatrixConversion.eulerAngles(2, 0, 2);

    //Eigen::Quaterniond
    Eigen::Quaterniond tmp = Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitZ())
     * Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitX())
     * Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitZ());

////////////////////////////////////////Comparing with KDL////////////////////////////////////////
#if KDL_FOUND==1
    KDL::Frame F;
    F.M = F.M.RPY(roll, pitch, yaw);
    std::cout << F.M(0, 0) << " " << F.M(0, 1) << " " << F.M(0, 2) << std::endl;
    std::cout << F.M(1, 0) << " " << F.M(1, 1) << " " << F.M(1, 2) << std::endl;
    std::cout << F.M(2, 0) << " " << F.M(2, 1) << " " << F.M(2, 2) << std::endl;

    double x, y, z, w;
    F.M.GetQuaternion(x, y, z, w);
    std::cout << "KDL Frame Quaternion:" << std::endl;
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
    std::cout << "z: " << z << std::endl;
    std::cout << "w: " << w << std::endl;
#endif


////////////////////////////////////////Comparing with KDL////////////////////////////////////////

    Eigen::Matrix3d rotation;
    rotation= eulerAnglesToRotationMatrix(roll, pitch,yaw);

    double 	txLeft, tyLeft, tzLeft;
    txLeft=-1;
    tyLeft=0.0;
    tzLeft=-4.0;

    Eigen::Affine3f t1;
    Eigen::Matrix4f M;
    Eigen::Vector3d translation;
    translation<<txLeft,tyLeft,tzLeft;

    M<<  rotation(0,0),rotation(0,1),rotation(0,2),translation(0,0)
     ,rotation(1,0),rotation(1,1),rotation(1,2),translation(1,0)
     ,rotation(2,0),rotation(2,1),rotation(2,2),translation(2,0)
     ,0,0,0,1;


    t1 = M;


    Eigen::Affine3d transform_2 = Eigen::Affine3d::Identity();


    // Define a translation of 2.5 meters on the x axis.
    transform_2.translation() << 2.5, 1.0, 0.5;

    // The same rotation matrix as before; tetha radians arround Z axis
    transform_2.rotate (yawAngle*pitchAngle *rollAngle );
    std::cout<<transform_2.matrix() <<std::endl;
    std::cout<<transform_2.translation()<<std::endl;
    std::cout<<transform_2.translation().x()<<std::endl;
    std::cout<<transform_2.translation().y()<<std::endl;
    std::cout<<transform_2.translation().z()<<std::endl;

}

void determiningRollPitchYawFromRotationMatrix()
{
    /*  http://planning.cs.uiuc.edu/node103.html

      |r11 r12 r13 |
      |r21 r22 r23 |
      |r31 r32 r33 |

      yaw: alpha=arctan(r21/r11)
      pitch: beta=arctan(-r31/sqrt( r32^2+r33^2 ) )
      roll: gamma=arctan(r32/r33)
  */
    double roll, pitch, yaw;
    roll = M_PI / 2;
    pitch = M_PI / 2;
    yaw = 0;
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d rotationMatrix = q.matrix();

    std::cout << "Rotation Matrix is:" << std::endl;
    std::cout << rotationMatrix << std::endl;

    std::cout << "roll is Pi/"
              << M_PI / atan2(rotationMatrix(2, 1), rotationMatrix(2, 2))
              << std::endl;
    std::cout << "pitch: Pi/"
              << M_PI / atan2(-rotationMatrix(2, 0),
                            std::pow(
                                rotationMatrix(2, 1) * rotationMatrix(2, 1) + rotationMatrix(2, 2) * rotationMatrix(2, 2),
                                0.5))
              << std::endl;
    std::cout << "yaw is Pi/"
              << M_PI / atan2(rotationMatrix(1, 0), rotationMatrix(0, 0))
              << std::endl;
}

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


Eigen::Matrix4f createAffinematrix(float a, float b, float c, Eigen::Vector3f trans)
{
    {
        Eigen::Transform<float, 3, Eigen::Affine> t;
        t = Eigen::Translation<float, 3>(trans);
        t.rotate(Eigen::AngleAxis<float>(a, Eigen::Vector3f::UnitX()));
        t.rotate(Eigen::AngleAxis<float>(b, Eigen::Vector3f::UnitY()));
        t.rotate(Eigen::AngleAxis<float>(c, Eigen::Vector3f::UnitZ()));
        return t.matrix();
    }


    {
    /*
    The difference between the first implementation and the second is like the difference between "Fix Angle" and "Euler Angle", you can
    https://www.youtube.com/watch?v=09xVHo1JudY
    */
        Eigen::Transform<float, 3, Eigen::Affine> t;
        t = Eigen::AngleAxis<float>(c, Eigen::Vector3f::UnitZ());
        t.prerotate(Eigen::AngleAxis<float>(b, Eigen::Vector3f::UnitY()));
        t.prerotate(Eigen::AngleAxis<float>(a, Eigen::Vector3f::UnitX()));
        t.pretranslate(trans);
        return t.matrix();
    }
}


int main()
{

}
