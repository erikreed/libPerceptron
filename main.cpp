//============================================================================
// Name        : MLP.cpp
// Author      : Erik Reed
// Version     : 
// Description : Implementation of an optimized, multithreaded, 
//				 single and multi-layer perceptron library in C++. 
//				 Fast neural network.
//============================================================================

#include <iostream>
using namespace std;

const double ETA = .45; // weight coefficient

template <typename T>
struct matrix {
	const unsigned int rows;
	const unsigned int cols;
	T *dat;

	matrix(const int rows, const int cols) : rows(rows), cols(cols) {
		dat = new T[rows*cols];
	}
	
	inline void set(const unsigned int row, const unsigned int col, const T &val) {
		dat[cols*row + col] = val;
	}

	inline T get(const unsigned int row, const unsigned int col) {
		return dat[cols*row + col];
	}

	~matrix() {
		delete[] dat;
	}
};

void SLP(const matrix<double> &in, const matrix<double> &out) {
	unsigned int numInputs = in.cols;
	unsigned int numVectors = in.rows;
	unsigned int numOutputs = out.cols;

	if (in.rows != out.rows)
		throw "number of test cases different between in/out...";

	// num neurons by num weights (+1 for input bias)
	matrix<double> weights(numInputs,numInputs+1);

	
}

void tests() {
	matrix<double> test1(4,2);
	matrix<double> test1out(4,1);

	// AND input
	test1.set(0,0,0);
	test1.set(0,1,0);
	test1.set(1,0,1);
	test1.set(1,1,0);
	test1.set(2,0,0);
	test1.set(2,1,1);
	test1.set(3,0,1);
	test1.set(4,1,1);

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

