//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#include "OptMLP.hpp"
#include "NeuralNetwork.hpp"

using namespace std;

Perceptron::~Perceptron() {
    if (weights == NULL)
        delete weights;
}

void Perceptron::train(Matrix<> &inputs, Matrix<> &outputs) {
    size_t numInputs = inputs.cols; // length of input vector
    size_t numVectors = inputs.rows; // i.e. numTargets
    size_t numOutputs = outputs.cols; // i.e. length of target vector

    if (inputs.rows != outputs.rows)
        throw "number of test cases different between in/out...";

    if (weights != NULL) {
        cout << "Perceptron: retraining... previous weights reset" << endl;
        delete weights;
    }
    // num neurons by num weights (+1 for input bias)
    weights = new Matrix<>(numOutputs, numInputs + 1);
    weights->randomize(10);

    char *activation = new char[numOutputs];
    for (size_t iter = 0; iter < MAX_ITERATIONS; iter++) {
        double diff = 0;
        for (size_t i = 0; i < numVectors; i++) {
            // randomize input order
            // NOTE: make sure there are somewhat equal numbers of classes,
            //       otherwise the network will be overly biased
            inputs.randomize_rows();
            // compute activations
            for (size_t j = 0; j < numOutputs; j++) {
                double sum = 0;
                for (size_t k = 0; k < numInputs; k++)
                    sum += weights->get(j, k) * inputs.get(i, k);
                sum += weights->get(j, numInputs) * -1; // input bias
                activation[j] = sum > 0 ? 1 : 0;
            }
            // update weights
            for (size_t j = 0; j < numOutputs; j++) {
                for (size_t k = 0; k < numInputs + 1; k++) {
                    double oldWeight = weights->get(j, k);
                    double newWeight = oldWeight
                            + ETA * (outputs.get(i, j) - activation[j])
                                    * inputs.get(i, k);
                    diff += oldWeight - newWeight;
                    weights->set(j, k, newWeight);
                }
            }
        }
        cout << weights << endl;
        if (diff == 0) //converged
            break;
    }

    delete[] activation;
}

