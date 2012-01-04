/*
 * OptMLP.hpp
 *
 *  Created on: Jan 4, 2012
 *      Author: Erik Reed
 */

#ifndef OPTMLP_HPP_
#define OPTMLP_HPP_

#include <iostream>
#include <stdlib.h>
#include <time.h>

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

#endif /* OPTMLP_HPP_ */
