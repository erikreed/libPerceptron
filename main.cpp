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
    cout << "Perceptron: testOR" << endl;
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
    double acc = p.test(test1,test1out);
    if (acc != 100)
        throw "testOR training failed";
    DataSet<char>* eval = p.evaluate(test1);
    if (!eval->equals(test1out))
        throw "testOR eval failed";
    delete eval;
    cout << endl;
}

void testAND() {
    cout << "Perceptron: testAND" << endl;
    DataSet<double> test1(2);
    DataSet<double> test1out(1);
    double test1new[] = {
            0,0,
            1,0,
            0,1,
            1,1
    };
    double test2new[] = {0,0,0,1};
    test1.addRows(test1new,4);
    test1out.addRows(test2new,4);

    Perceptron p;
    // training data = testing data
    p.train(test1, test1out, true);
    double acc = p.test(test1,test1out);
    if (acc != 100)
        throw "testAND training failed";
    DataSet<char>* eval = p.evaluate(test1);
    if (!eval->equals(test1out))
        throw "testAND eval failed";
    delete eval;
    cout << endl;
}

int main(int argc, char** args) {
    cout << "--- Testing ---" << endl;
    testOR();
    testAND();

    return 0;
}

