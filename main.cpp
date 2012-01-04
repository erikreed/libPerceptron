//============================================================================
// Name        : MLP.cpp
// Author      : Erik Reed
// Version     : 
// Description : Implementation of an optimized, multithreaded, 
//               single and multi-layer perceptron library in C++. 
//               Fast neural network.
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

const double ETA = .2; // weight coefficient
// typical 0.1 < ETA < 0.4
const size_t MAX_ITERATIONS = 15;

template <typename T>
struct matrix {
	const size_t rows;
	const size_t cols;
	T *dat;

	matrix(const size_t rows, const size_t cols) : rows(rows), cols(cols) {
		dat = new T[rows*cols];
	}

	inline void set(const size_t row, const size_t col, const T &val) {
		dat[cols*row + col] = val;
	}

	inline T get(const size_t row, const size_t col) {
		return dat[cols*row + col];
	}

	void randomize() {
		srand (time(NULL));
		for (size_t i=0; i<rows*cols; i++) {
			dat[i] = rand() / RAND_MAX - .5; // rand in [-.5, .5]
		}
	}

	~matrix() {
		delete[] dat;
	}
};

// single layer perceptron with threshold activation function
void SLP(matrix<double> &in, matrix<double> &out) {
	size_t numInputs = in.cols; // length of input vector
	size_t numVectors = in.rows; // i.e. numTargets
	size_t numOutputs = out.cols; // i.e. length of target vector

	if (in.rows != out.rows)
		throw "number of test cases different between in/out...";

	// num neurons by num weights (+1 for input bias)
	matrix<double> weights(numOutputs,numInputs+1);
	weights.randomize();

	char *activation = new char[numOutputs];
	for (size_t iter=0; iter<MAX_ITERATIONS; iter++) {
		double diff = 0;
		for (size_t i=0; i<numVectors; i++) {
			// compute activations
			for (size_t j=0; j<numOutputs; j++) {
				double sum = 0;
				for (size_t k=0; k<numInputs; k++)
					sum += weights.get(j,k) * in.get(i,k);
				sum += weights.get(j,numInputs) * -1; // input bias
				activation[j] = sum > 0 ? 1 : 0;
			}
			// update weights
			for (size_t j=0; j<numOutputs; j++) {
				for (size_t k=0; k<numInputs+1; k++) {
					double oldWeight = weights.get(j,k);
					double newWeight = oldWeight + ETA*(out.get(i,j)-activation[j])*in.get(i,k);
					diff += oldWeight - newWeight;
					weights.set(j,k,newWeight);
				}
			}
			cout << "Iteration: " << iter << endl;
			if (diff==0) //converged
				break;
		}
	}
	delete[] activation;
}

void tests() {
	matrix<double> test1(4,2);
	matrix<double> test1out(4,1);

	// OR input
	// TODO: more optimal setting
	test1.set(0,0,0);
	test1.set(0,1,0);
	test1.set(1,0,1);
	test1.set(1,1,0);
	test1.set(2,0,0);
	test1.set(2,1,1);
	test1.set(3,0,1);
	test1.set(3,1,1);

	test1out.set(0,0,0);
	test1out.set(1,0,1);
	test1out.set(2,0,1);
	test1out.set(3,0,1);

	SLP(test1,test1out);
}

int main(int argc, char** args){

	tests();
	return 0;
}

