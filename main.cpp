//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded, 
//               single and multi-layer perceptron library in C++. 
//               Fast neural network.
//============================================================================

#include "OptMLP.hpp"
#include <typeinfo>

using namespace std;

void testOR(NeuralNetwork &NN) {
    cout << typeid(NN).name() << ": testOR" << endl;
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


    // training data = testing data
    NN.train(test1, test1out);
    double acc = NN.test(test1,test1out);
    if (acc != 100)
        throw "testOR training failed";
    DataSet<char>* eval = NN.evaluate(test1);
    if (!eval->equals(test1out))
        throw "testOR eval failed";
    delete eval;
    cout << endl;
}

void testAND(NeuralNetwork &NN) {
    cout << typeid(NN).name() << ": testAND" << endl;
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

    // training data = testing data
    NN.train(test1, test1out);
    double acc = NN.test(test1,test1out);
    if (acc != 100)
        throw "testAND training failed";
    DataSet<char>* eval = NN.evaluate(test1);
    if (!eval->equals(test1out))
        throw "testAND eval failed";
    delete eval;
    cout << endl;
}

int main(int argc, char** args) {
    cout << "--- Testing ---" << endl;
    Perceptron p1,p2;
    testOR(p1);
    testAND(p2);

    MLPerceptron mlp1,mlp2;
    testOR(mlp1);
    testAND(mlp2);

    return 0;
}

