//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#ifndef NEURALNETWORK_HPP_
#define NEURALNETWORK_HPP_

#include "DataSet.hpp"

class NeuralNetwork {
public:
    virtual void train(DataSet<> &inputs, DataSet<> &outputs) = 0;
    virtual DataSet<double>* evaluate(DataSet<> &inputs) = 0;
    //virtual void verify(Matrix<> &inputs, Matrix<> &outputs) = 0;
    virtual double test(DataSet<> &inputs, DataSet<> &outputs) = 0;

    //virtual Matrix<> readDataFromFile(char* path);
    virtual ~NeuralNetwork() {
    }
};

// single layer perceptron with threshold activation function
class Perceptron: public NeuralNetwork {
    DataSet<> *weights;
    void recall(size_t numOutputs, size_t numInputs, DataSet<> &inputs,
            size_t i, double *activation);

public:

    // weight coefficient
    // typical value: 0.1 < ETA < 0.4
    double eta;
    size_t max_iterations;

    void train(DataSet<> &inputs, DataSet<> &outputs);
    void train(DataSet<> &inputs, DataSet<> &outputs, bool randomize_rows);
    double test(DataSet<> &inputs, DataSet<> &outputs);
    DataSet<double>* evaluate(DataSet<> &inputs);
    ~Perceptron();
    Perceptron();
};

class MLPerceptron: public NeuralNetwork {
    DataSet<> **weights;
    void recallLayer(size_t outputDim, size_t inputDim,
            double *inputs, double *activation, DataSet<> *weight);
    void recall(size_t numOutputs, size_t numInputs, DataSet<> &inputs,
            size_t i, DataSet<double> &activation);
    void clean();
    double round(double d);
public:

    // weight coefficient
    // typical value: 0.1 < ETA < 0.4
    double eta;
    size_t max_iterations;
    const size_t numHiddenLayers;
    double beta; // sigmoid function constant
    double tol; // convergence tolerance
    double alpha; //momentum constant

    void train(DataSet<> &inputs, DataSet<> &outputs);
    void train(DataSet<> &inputs, DataSet<> &outputs,
            bool randomize_rows);
    double test(DataSet<> &inputs, DataSet<> &outputs);
    DataSet<double>* evaluate(DataSet<> &inputs);
    ~MLPerceptron();
    MLPerceptron();
    MLPerceptron(size_t numHiddenLayers);
};

class RBFPerceptron: NeuralNetwork {

};

#endif /* NEURALNETWORK_HPP_ */
