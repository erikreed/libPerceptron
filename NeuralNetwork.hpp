//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#ifndef NEURALNETWORK_HPP_
#define NEURALNETWORK_HPP_

#include "OptMLP.hpp"

class NeuralNetwork {
public:
    virtual void train(Matrix<> &inputs, Matrix<> &outputs) = 0;
    //virtual void evaluate(Matrix<> &inputs) = 0;
    //virtual void verify(Matrix<> &inputs, Matrix<> &outputs) = 0;
    virtual void test(Matrix<> &inputs, Matrix<> &outputs) = 0;

    //virtual Matrix<> readDataFromFile(char* path);
    virtual ~NeuralNetwork() {};

};

// single layer perceptron with threshold activation function
class Perceptron : NeuralNetwork {
    Matrix<> *weights;

public:

    void train(Matrix<> &inputs, Matrix<> &outputs);
    void test(Matrix<> &inputs, Matrix<> &outputs);
    ~Perceptron();
    Perceptron();
};

class MLPerceptron: NeuralNetwork {
};
class RBFPerceptron: NeuralNetwork {
};

#endif /* NEURALNETWORK_HPP_ */
