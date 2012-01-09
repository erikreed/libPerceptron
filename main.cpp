//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded, 
//               single and multi-layer perceptron library in C++. 
//               Fast neural network.
//============================================================================

#include "OptMLP.hpp"
#include "DataUtils.hpp"
using namespace std;

void tests() {
    Matrix<double> test1(4, 2);
    Matrix<double> test1out(4, 1);

    // OR input
    // TODO: more optimal setting
    test1.set(0, 0, 0);
    test1.set(0, 1, 0);
    test1.set(1, 0, 1);
    test1.set(1, 1, 0);
    test1.set(2, 0, 0);
    test1.set(2, 1, 1);
    test1.set(3, 0, 1);
    test1.set(3, 1, 1);

    test1out.set(0, 0, 0);
    test1out.set(1, 0, 1);
    test1out.set(2, 0, 1);
    test1out.set(3, 0, 1);

    Perceptron p;
    // training data = testing data
    p.train(test1, test1out);
    p.test(test1,test1out);
}

int main(int argc, char** args) {

    tests();
    return 0;
}

