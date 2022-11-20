#!/bin/sh
set -e

# Build libwallaby
git clone https://github.com/kipr/libwallaby
cd libwallaby
rm -rf tests
mkdir build
cd build
# cmake .. -DCMAKE_CXX_FLAGS=-I/usr/include/opencv4 -Dbuild_python=ON -Dwith_vision_support=OFF -DBUILD_DOCUMENTATION=OFF
cmake .. -Dbuild_python=OFF -Dwith_vision_support=OFF -DBUILD_DOCUMENTATION=OFF
make -j8 install

# Build python wrapper
cd ..
mv /kipr_c.i .
swig -python -I/libwallaby/include/kipr kipr_c.i
gcc -shared -fpic -o _kipr.so \
    -I/usr/include/python3.10 -I/usr/include/x86_64-linux-gnu/python3.9 \
    -I/libwallaby/include/kipr \
    kipr_c_wrap.c -lpython3.10
mv kipr.py _kipr.so lib/
