https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build 
cd build



cmake -DCMAKE_CXX_FLAGS=-std=c++1z -DCMAKE_BUILD_TYPE=Release  -DCMAKE_INSTALL_PREFIX:PATH=~/usr .. && make -j8 all install 
