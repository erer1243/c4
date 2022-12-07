#!/bin/sh
set -e
wget https://github.com/PascalPons/connect4/releases/download/book/7x6.book &
git clone https://github.com/PascalPons/connect4
cd connect4
git apply ../c4solver.patch
make -j4
wait
mv ../7x6.book .
