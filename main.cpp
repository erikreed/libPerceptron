//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded, 
//               single and multi-layer perceptron library in C++. 
//               Fast neural network.
//============================================================================

#include "OptMLP.hpp"

using namespace std;

void testOR() {
    DataSet<double> test1(2);
    DataSet<double> test1out(1);
    double test1new[] = {
            0,0,
            1,0,
            0,1,
            1,1
    };
    double test2new[] = {0,1,1,1};
    test1.addRows(test1new,4);
    test1out.addRows(test2new,4);

    Perceptron p;
    // training data = testing data
    p.train(test1, test1out, true);
    p.test(test1,test1out);
}

int main(int argc, char** args) {

    testOR();
    return 0;
}

