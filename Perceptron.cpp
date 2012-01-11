//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#include "OptMLP.hpp" //TODO: get rid of this
#include "NeuralNetwork.hpp"

using namespace std;

// TODO: allow linear output instead of discrete (template out char)
// TODO: test this
DataSet<char>* Perceptron::evaluate(DataSet<> &inputs) {
    // weight row dimension corresponds to output vector length
    DataSet<char> *outputs = new DataSet<char>(weights->rows);
    char *activation = new char[weights->rows];

    for (size_t i = 0; i < inputs.rows; i++) {
        recall(weights->rows, inputs.cols, inputs, i, activation);
        outputs->addRow(activation);
    }
    delete activation;
    return outputs;
}

inline void Perceptron::recall(size_t numOutputs, size_t numInputs,
        DataSet<> &inputs, size_t input_row, char *activation) {
    // compute activations
    for (size_t j = 0; j < numOutputs; j++) {
        double sum = 0;
        for (size_t k = 0; k < numInputs; k++)
            sum += weights->get(j, k) * inputs.get(input_row, k);

        sum += weights->get(j, numInputs) * -1; // input bias
        activation[j] = sum > 0 ? 1 : 0;
    }
}

double Perceptron::test(DataSet<> &inputs, DataSet<> &outputs) {
    size_t numInputs = inputs.cols; // length of input vector
    size_t numVectors = inputs.rows; // i.e. numTargets
    size_t numOutputs = outputs.cols; // i.e. length of target vector

    DEBUG_MSG("Perceptron::test");
    DEBUG_MSG("Number of inputs: " << numVectors);
    DEBUG_MSG("Target vector length: " << outputs.cols << endl);

    if (inputs.rows != outputs.rows)
        throw "number of test cases different between in/out: ";

    if (weights == NULL)
        throw "Perceptron has not been trained yet.";

    char *activation = new char[numOutputs];

    size_t testsCorrect = 0;

    for (size_t i = 0; i < numVectors; i++) {
        // compute activations
        recall(numOutputs, numInputs, inputs, i, activation);
        // compute accuracy
        bool correct = true;
        for (size_t j = 0; j < numOutputs; j++) {
            if (activation[j] != outputs.get(i, j)) {
                correct = false;
                break;
            }
        }
        if (correct)
            testsCorrect++;
    }
    delete activation;
    double accuracy = 1.0 - ((double) (numVectors - testsCorrect)) / numVectors;
    accuracy *= 100;
//    cout << "Total classifications: " << numVectors << endl;
//    cout << "Correct classifications: " << testsCorrect << endl;
//    cout << "Incorrect classifcations: " << numVectors - testsCorrect << endl;
    cout << "Classification Accuracy: " << testsCorrect << "/" << numVectors
            << " = " << accuracy << "%" << endl;
    return accuracy;
}

Perceptron::Perceptron() {
    weights = NULL;
}

Perceptron::~Perceptron() {
    if (weights == NULL)
        delete weights;
}

void Perceptron::train(DataSet<> &inputs, DataSet<> &outputs) {
    Perceptron::train(inputs, outputs, true);
}

void Perceptron::train(DataSet<> &inputs, DataSet<> &outputs,
        bool randomize_rows) {

    size_t numInputs = inputs.cols; // length of input vector
    size_t numVectors = inputs.rows; // i.e. numTargets
    size_t numOutputs = outputs.cols; // i.e. length of target vector

    DEBUG_MSG("Perceptron::train");
    DEBUG_MSG("Number of inputs: " << numVectors);
    DEBUG_MSG("Target vector length: " << outputs.cols << endl);

    if (inputs.rows != outputs.rows)
        throw "number of test cases different between in/out: ";

    if (weights != NULL) {
        cout << "Perceptron: retraining... previous weights reset" << endl;
        delete weights;
    }
    // num neurons by num weights (+1 for input bias)
    weights = new DataSet<>(numOutputs, numInputs + 1);
    weights->randomize(10);

    char *activation = new char[numOutputs];
    size_t iter;
    for (iter = 0; iter < 15; iter++) {
        double diff = 0;
        if (randomize_rows)
            DataSet<>::randomize_rows(inputs,outputs);
        for (size_t i = 0; i < numVectors; i++) {
            // NOTE: make sure there are somewhat equal numbers of classes,
            //       otherwise the network will be overly biased

            // compute activations
            recall(numOutputs, numInputs, inputs, i, activation);
            // update weights
            for (size_t j = 0; j < numOutputs; j++) {
                for (size_t k = 0; k < numInputs; k++) {
                    double oldWeight = weights->get(j, k);
                    double newWeight = oldWeight
                            + ETA * (outputs.get(i, j) - activation[j])
                                    * inputs.get(i, k);
                    diff += oldWeight - newWeight > 0 ?
                            oldWeight - newWeight : -(oldWeight - newWeight);
                    weights->set(j, k, newWeight);
                }

                // bias term
                double oldWeight = weights->get(j, numInputs);
                double newWeight = oldWeight
                        + ETA * (outputs.get(i, j) - activation[j]) * -1;
                diff += oldWeight - newWeight > 0 ?
                        oldWeight - newWeight : -(oldWeight - newWeight);
                weights->set(j, numInputs, newWeight);
            }
        }

        if (diff == 0) //converged
            break;
    }
    if (iter != MAX_ITERATIONS)
        cout << "Converged in " << iter << " iterations." << endl;
    else
        cout << "Max iterations reached: " << iter << endl;
    delete[] activation;
}

