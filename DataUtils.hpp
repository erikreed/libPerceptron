//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#ifndef DATAUTILS_HPP_
#define DATAUTILS_HPP_

#include <iostream>
#include <stdlib.h>
#include <time.h>

template<class T = double>
class Matrix {
protected:
    T *dat;
public:
    const std::size_t rows;
    const std::size_t cols;

    explicit Matrix(const std::size_t rows, const std::size_t cols);
    explicit Matrix(const Matrix<T> &m);
    ~Matrix();
    void printRow(std::size_t row);
    T* getRow(std::size_t row);
    T& get(const std::size_t row, const std::size_t col);
    void set(const std::size_t row, const std::size_t col, const T &val);
    void randomize_rows();
    void randomize();
    void randomize(double normalize);

    template<class K>
    friend std::ostream& operator<<(std::ostream& cout, Matrix<K> const &m);
};

#endif /* DATAUTILS_HPP_ */
