//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer MLPerceptron library in C++.
//               Fast neural network.
//============================================================================

#include "NeuralNetwork.hpp"
#include "math.h"

using namespace std;

const int NORMALIZE = 10;
const double INIT_ETA = 0.2;
const size_t INIT_MAX_ITER = 25;
const size_t INIT_NUM_HIDDEN = 1;
const double INIT_BETA = 1;

MLPerceptron::MLPerceptron(size_t numHiddenLayers = INIT_NUM_HIDDEN) :
        eta(INIT_ETA), max_iterations(INIT_MAX_ITER), numHiddenLayers(
                numHiddenLayers), beta(INIT_BETA) {
    if (numHiddenLayers < 1)
        throw "Invalid number of MLP hidden layers";
    weights = NULL;
}

// TODO: allow linear output instead of discrete (template out char)
DataSet<char>* MLPerceptron::evaluate(DataSet<> &inputs) {
    // weight row dimension corresponds to output vector length
    DataSet<char> *outputs = new DataSet<char>(5); //(weights->rows);
//    char *activation = new char[weights->rows];
//
//    for (size_t i = 0; i < inputs.rows; i++) {
//        recall(weights->rows, inputs.cols, inputs, i, activation);
//        outputs->addRow(activation);
//    }
//    delete[] activation;
    return outputs;
}

void MLPerceptron::recall(size_t outputDim, size_t inputDim, DataSet<> &inputs,
        size_t input_row, DataSet<double> &activation) {
    // first layer
    recallLayer(outputDim, inputDim, inputs.getRow(input_row),
            activation.getRow(0), weights[0]);
    // rest of hidden layers and output layer
    for (size_t i = 1; i < numHiddenLayers + 1; i++)
        recallLayer(outputDim, outputDim, inputs.getRow(input_row),
                activation.getRow(i), weights[i]);
}

void MLPerceptron::recallLayer(size_t outputDim, size_t inputDim,
        double *inputs, double *activation, DataSet<> *weight) {
    for (size_t j = 0; j < outputDim; j++) {
        double sum = 0;
        for (size_t k = 0; k < inputDim; k++)
            sum += weight->get(j, k) * inputs[k];

        sum += weight->get(j, inputDim) * -1; // input bias
        //activation[j] = sum > 0 ? 1 : 0;
        activation[j] = 1.0 / (exp(-beta * sum));
    }
}

double MLPerceptron::test(DataSet<> &inputs, DataSet<> &outputs) {
//    size_t inputDim = inputs.cols; // length of input vector
//    size_t numVectors = inputs.rows; // i.e. number of target vectors
//    size_t targetDim = outputs.cols; // i.e. length of target vector
//
//    if (inputs.rows != outputs.rows)
//        throw "number of test cases different between in/out: ";
//
//    if (weights == NULL)
//        throw "MLPerceptron has not been trained yet.";
//
//    char *activation = new char[targetDim];
//
//    size_t testsCorrect = 0;
//
//    for (size_t i = 0; i < numVectors; i++) {
//        // compute activations
//        recall(targetDim, inputDim, inputs, i, activation);
//        // compute accuracy
//        bool correct = true;
//        for (size_t j = 0; j < targetDim; j++) {
//            if (activation[j] != outputs.get(i, j)) {
//                correct = false;
//                break;
//            }
//        }
//        if (correct)
//            testsCorrect++;
//    }
//    delete[] activation;
//    double accuracy = 1.0 - ((double) (numVectors - testsCorrect)) / numVectors;
//    accuracy *= 100;
//    cout << "Classification Accuracy: " << testsCorrect << "/" << numVectors
//            << " = " << accuracy << "%" << endl;
//    return accuracy;
    return 5;
}

void MLPerceptron::clean() {
    if (weights != NULL) {
        for (size_t i = 0; i < numHiddenLayers + 1; i++)
            delete weights[i];
        delete[] weights;
    }
}

MLPerceptron::~MLPerceptron() {
    clean();
}

void MLPerceptron::train(DataSet<> &inputs, DataSet<> &outputs) {
    MLPerceptron::train(inputs, outputs, true);
}

void MLPerceptron::train(DataSet<> &inputs, DataSet<> &outputs,
        bool randomize_rows) {

    size_t inputDim = inputs.cols; // length of input vector
    size_t numVectors = inputs.rows; // i.e. number of target vectors
    size_t targetDim = outputs.cols; // i.e. length of target vector

    if (inputs.rows != outputs.rows)
        throw "number of test cases different between in/out: ";

    if (weights != NULL) {
        cout << "MLPerceptron: retraining... previous weights reset" << endl;
        clean();
    }
    // num neurons by num weights (+1 for input bias)
    weights = new DataSet<>*[numHiddenLayers + 1];
    weights[0] = new DataSet<>(targetDim, inputDim + 1);
    weights[0]->randomize(NORMALIZE);
    for (size_t i = 1; i < numHiddenLayers + 1; i++) {
        weights[i] = new DataSet<>(targetDim, targetDim + 1);
        weights[i]->randomize(NORMALIZE);
    }
    //(targetDim, inputDim + 1);

    DataSet<double> activation(targetDim, numHiddenLayers + 1);
    DataSet<double> error(targetDim, numHiddenLayers + 1);

    size_t iter;

    for (iter = 0; iter < 15; iter++) {
        if (randomize_rows)
            DataSet<>::randomize_rows(inputs, outputs);
        double diff = 0;
        for (size_t i = 0; i < numVectors; i++) {
            // NOTE: make sure there are somewhat equal numbers of classes,
            //       otherwise the network will be overly biased

            // compute activations (forward)
            recall(targetDim, inputDim, inputs, i, activation);
            // compute error in output
            for (size_t j = 0; j < targetDim; j++) {
                double calc = activation.get(numHiddenLayers, j);
                double actual = outputs.get(i, j);
                double error_calc = (actual - calc) * calc * (1 - calc);
                error.set(numHiddenLayers, j, error_calc);
            }
            // compute error in hidden layers (backwards propagation)
            for (size_t w = numHiddenLayers - 1; w >= 0; w++) {
                DataSet<double> *weight = weights[i];
                for (size_t j = 0; j < targetDim; j++) {
                    double sum = 0;
                    for (size_t k = 0; k < targetDim; k++) {
                        sum += weight->get(j,k);
                    }
                    sum += weight->get(j,targetDim);
                    double calc = activation.get(j, w);
                    double actual = outputs.get(i, j);
                    double error_calc = (actual - calc) * calc * (1 - calc);
                    error.set(numHiddenLayers, j, error_calc);
                }
            }
        }
        if (diff == 0) //converged
            break;
    }
    if (iter != max_iterations)
        cout << "Converged in " << iter << " iterations." << endl;
    else
        cout << "Max iterations reached: " << iter << endl;
}

