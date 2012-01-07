//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#ifndef OPTMLP_HPP_
#define OPTMLP_HPP_

#include <iostream>
#include <stdlib.h>
#include <time.h>

// typical 0.1 < ETA < 0.4
const double ETA = .2; // weight coefficient
const size_t MAX_ITERATIONS = 15;

template<class T = double>
class Matrix {
protected:
    T *dat;
public:
    const size_t rows;
    const size_t cols;

    Matrix(const size_t rows, const size_t cols) :
            rows(rows), cols(cols) {
        dat = new T[rows * cols];
    }

    Matrix(const Matrix<T> &m) {
        this(m.rows, m.cols);
        for (size_t i = 0; i < rows * cols; i++)
            dat[i] = m.dat[i];
    }

    void printRow(size_t row) {
        T* rowData = getRow(row);
        for (size_t i = 0; i < rows; i++)
            std::cout << rowData[i] << " ";
        std::cout << std::endl;
    }

    T* getRow(size_t row) {
        return &dat[row * rows];
    }

    void set(const size_t row, const size_t col, const T &val) {
        dat[cols * row + col] = val;
    }

    T& get(const size_t row, const size_t col) {
        return dat[cols * row + col];
    }

    void randomize_rows() {
        for (size_t i = 0; i < rows; i++) {
            size_t row = (rand() / (double) RAND_MAX) * rows;
            if (i != row) {
                for (size_t j = 0; j < cols; j++)
                    std::swap(get(i, j), get(row, j));
            }
        }
    }

    void randomize() {
        randomize(1);
    }

    void randomize(double normalize) {
        srand(time(NULL));
        for (size_t i = 0; i < rows * cols; i++) {
            dat[i] = (rand() / (double) RAND_MAX - .5) * 2.0 / normalize;
        }
    }

    ~Matrix() {
        delete[] dat;
    }

    friend std::ostream& operator<<(std::ostream& cout, Matrix& m) {
        for (size_t i = 0; i < m.rows * m.cols; i++) {
            if (i % m.cols == 0 && i != 0)
                cout << std::endl;
            cout << m.dat[i] << " ";
        }
        cout << std::endl;
        return cout;
    }

};

#endif /* OPTMLP_HPP_ */
