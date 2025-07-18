#!/usr/bin/env bash

echo "Installing blocktrix..."
g++ -O2 -std=c++11 test_blocktrix.cpp blocktrix_solver.cpp -L/opt/homebrew/Cellar/lapack/3.12.1/lib -I/opt/homebrew/Cellar/lapack/3.12.1/include -llapack -o test_blocktrix
echo "Installation complete."

echo "Running test_blocktrix..."
if [ ! -f ./test_blocktrix ]; then
    echo "Error: test_blocktrix not found. Please check the installation."
    exit 1
fi
./test_blocktrix
echo "Testing completed."
