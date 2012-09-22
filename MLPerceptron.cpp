//============================================================================
// Name        :
// Author      : Erik Reed
// Description :
//============================================================================

#include "NeuralNetwork.hpp"
#include "math.h"

using namespace std;

const double INIT_ETA = 0.2;
const size_t INIT_MAX_ITER = 5;
const size_t INIT_NUM_HIDDEN = 1;
const double INIT_BETA = 1;
const double INIT_TOL = 1e-4;
const double INIT_ALPHA = 0;

MLPerceptron::MLPerceptron() :
        eta(INIT_ETA), max_iterations(INIT_MAX_ITER), numHiddenLayers(
                INIT_NUM_HIDDEN), beta(INIT_BETA), tol(INIT_TOL), alpha(
                INIT_ALPHA) {
    if (numHiddenLayers < 1)
        throw "Invalid number of MLP hidden layers";
    weights = NULL;
}

MLPerceptron::MLPerceptron(size_t numHiddenLayers) :
        eta(INIT_ETA), max_iterations(INIT_MAX_ITER), numHiddenLayers(
                numHiddenLayers), beta(INIT_BETA), tol(INIT_TOL), alpha(
                INIT_ALPHA) {
    if (numHiddenLayers < 1)
        throw "Invalid number of MLP hidden layers";
    weights = NULL;
}

// TODO: allow discrete output instead of linear (template out double)
DataSet<double>* MLPerceptron::evaluate(DataSet<> &inputs) {
    // weight row dimension corresponds to output vector length
    if (weights == NULL)
        throw "MLPerceptron has not been trained yet.";
    if (inputs.cols != weights[0]->rows)
        throw "Number of inputs different from training data";
    DataSet<double> *outputs = new DataSet<double>(weights[1]->rows);
    DataSet<double> activation(numHiddenLayers + 1, weights[1]->rows);

    for (size_t i = 0; i < inputs.rows; i++) {
        recall(weights[1]->rows, inputs.cols, inputs, i, activation);
        outputs->addRow(activation.getRow(numHiddenLayers));
    }
    return outputs;
}

void MLPerceptron::recall(size_t outputDim, size_t inputDim, DataSet<> &inputs,
        size_t input_row, DataSet<double> &activation) {
    // dim of activation = [numHiddenLayers+1][targetDim]
    // first layer
    recallLayer(outputDim, inputDim, inputs.getRow(input_row),
            activation.getRow(0), weights[0]);
    // rest of hidden layers and output layer
    for (size_t i = 1; i < numHiddenLayers + 1; i++) {
        recallLayer(outputDim, outputDim, activation.getRow(i - 1),
                activation.getRow(i), weights[i]);
    }
}

void MLPerceptron::recallLayer(size_t outputDim, size_t inputDim,
        double *inputs, double *activation, DataSet<> *weight) {
    for (size_t j = 0; j < outputDim; j++) {
        double sum = 0;
        for (size_t k = 0; k < inputDim; k++)
            sum += weight->get(j, k) * inputs[k];
        sum += weight->get(j, inputDim) * -1; // input bias
        activation[j] = 1.0 / (1.0 + exp(-beta * sum));
    }
}

double MLPerceptron::test(DataSet<> &inputs, DataSet<> &outputs) {
    size_t inputDim = inputs.cols; // length of input vector
    size_t numVectors = inputs.rows; // i.e. number of target vectors
    size_t targetDim = outputs.cols; // i.e. length of target vector

    if (inputs.rows != outputs.rows)
        throw "number of test cases different between in/out: ";
    if (weights == NULL)
        throw "MLPerceptron has not been trained yet.";
    //    if (inputDim != weights[0]->rows)
    //        throw "Number of inputs different from training data";

    size_t testsCorrect = 0;
    DataSet<double> activation(numHiddenLayers + 1, weights[1]->rows);

    for (size_t i = 0; i < inputs.rows; i++) {
        recall(weights[1]->rows, inputDim, inputs, i, activation);
        bool correct = true;
        for (size_t j = 0; j < targetDim; j++) {
            cout << activation.get(numHiddenLayers, j) << ", "
                    << outputs.get(i, j) << endl;
            if (activation.get(numHiddenLayers, j) != outputs.get(i, j)) {
                correct = false;
                break;
            }
        }
        if (correct)
            testsCorrect++;
    }
    double accuracy = 1.0 - ((double) (numVectors - testsCorrect)) / numVectors;
    accuracy *= 100;
    cout << "Classification Accuracy: " << testsCorrect << "/" << numVectors
            << " = " << accuracy << "%" << endl;
    return accuracy;
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
    // using "uniform learning" initialization
    weights[0]->randomize(sqrt(inputDim));
    for (size_t i = 1; i < numHiddenLayers + 1; i++) {
        weights[i] = new DataSet<>(targetDim, targetDim + 1);
        weights[i]->randomize(sqrt(inputDim));
    }
    //TODO: get rid of old weights for momentum
    DataSet<>** old_weights = new DataSet<>*[numHiddenLayers + 1];
    old_weights[0] = new DataSet<>(targetDim, inputDim + 1);
    for (size_t i = 1; i < numHiddenLayers + 1; i++)
        old_weights[i] = new DataSet<>(*weights[i]);

    // layout = [layer][target]
    //
    //        ^
    // output |
    //        |
    // layers |
    //        |
    // hidden *------------------------>
    //           targets  -> targetDim
    DataSet<double> activation(numHiddenLayers + 1, targetDim);
    DataSet<double> error(numHiddenLayers + 1, targetDim);

    size_t iter;

    for (iter = 0; iter < max_iterations; iter++) {
        if (randomize_rows)
            DataSet<>::randomize_rows(inputs, outputs);
        double diff = 0;
        for (size_t i = 0; i < numVectors; i++) {
            // NOTE: make sure there are somewhat equal numbers of classes,
            //       otherwise the network will be overly biased

            // compute activations (forwards phase)
            recall(targetDim, inputDim, inputs, i, activation);
//            cout << "weight0\n" << *weights[0] << "\nend weights0" << endl;
//            cout << "weight1\n" << *weights[1] << "\nend weights1" << endl;
//            cout << "activation\n" << activation << "\nend activation" << endl;
            // compute error in output
            for (size_t j = 0; j < targetDim; j++) {
                double calc = activation.get(numHiddenLayers, j);
                double actual = outputs.get(i, j);
                double error_calc = (actual - calc) * calc * (1 - calc);
                //double error_calc = (calc - actual) * calc * (1 - calc);
                error.set(numHiddenLayers, j, error_calc);
            }
            // backwards propagation -- compute error in hidden layers
            for (size_t w = numHiddenLayers - 1; w > 0; w--) {
                DataSet<double> *weight = weights[w];
                for (size_t j = 0; j < targetDim; j++) {
                    double sum = 0;
                    for (size_t k = 0; k < targetDim + 1; k++)
                        sum += weight->get(j, k) * error.get(w + 1, j);
                    double calc = activation.get(w, j);
                    double error_calc = calc * (1 - calc) * sum;
                    error.set(w, j, error_calc);
                }
            }

            // 0th hidden weights (inputs)
            DataSet<double> *weight0 = weights[0];
            for (size_t j = 0; j < targetDim; j++) {
                double sum = 0;
                for (size_t k = 0; k < inputDim + 1; k++)
                    sum += weight0->get(j, k) * error.get(1, j);
                double calc = activation.get(0, j);
                double error_calc = calc * (1 - calc) * sum;
                error.set(0, j, error_calc);
            }
//            cout << "error\n" << error << endl;
            // backwards propagation -- update weights
            for (size_t w = numHiddenLayers; w > 0; w--) {
                DataSet<double> *weight = weights[w];
                for (size_t j = 0; j < targetDim; j++) {
                    for (size_t k = 0; k < targetDim + 1; k++) {
                        double currentWeight = weight->get(j, k);
                        double newWeight = currentWeight
                                + eta * error.get(w, j) * activation.get(w, j);
                        double momentum =
                                (newWeight - old_weights[w]->get(j, k)) * alpha;
                        weight->set(j, k, newWeight + momentum);
                        diff += pow(currentWeight - newWeight, 2);
                    }
                }
            }
            // inputs weight
            for (size_t j = 0; j < targetDim; j++) {
                for (size_t k = 0; k < inputDim + 1; k++) {
                    double currentWeight = weight0->get(j, k);
                    double newWeight = currentWeight
                            + eta * error.get(0, j) * activation.get(0, j);
                    double momentum = (newWeight - old_weights[0]->get(j, k))
                            * alpha;
                    weight0->set(j, k, newWeight + momentum);
                    diff += pow(currentWeight - newWeight, 2);
                }
            }
            for (size_t i = 0; i < numHiddenLayers + 1; i++)
                *old_weights[i] = *weights[i];
        }
        cout << "diff: " << diff << endl;
        if (diff <= tol) //converged
            break;
    }
    if (iter != max_iterations)
        cout << "Converged in " << iter << " iterations." << endl;
    else
        cout << "Max iterations reached: " << iter << endl;
    for (size_t i = 0; i < numHiddenLayers + 1; i++)
        delete old_weights[i];
    delete[] old_weights;
}

// TODO: maybe use for double->char for discretization
inline double MLPerceptron::round(double d) {
    return floor(d + 0.5);
}
