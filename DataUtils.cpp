//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#include "DataUtils.hpp"
#include <sstream>

using namespace std;

template<class T>
Matrix<T>::Matrix(const size_t rows, const size_t cols) :
        rows(rows), cols(cols) {
    dat = new T[rows * cols];
}

template<class T>
Matrix<T>::Matrix(const Matrix<T> &m) :
        rows(m.rows), cols(m.cols) {
    for (size_t i = 0; i < rows * cols; i++)
        dat[i] = m.dat[i];
}

template<class T>
void Matrix<T>::printRow(size_t row) {
    T* rowData = getRow(row);
    for (size_t i = 0; i < rows; i++)
        std::cout << rowData[i] << " ";
}

template<class T>
string Matrix<T>::sPrintRow(size_t row) {
    T* rowData = getRow(row);
    std::stringstream ss;
    for (size_t i = 0; i < rows; i++)
        ss << rowData[i] << " ";
    return ss.str();
}

template<class T>
inline T* Matrix<T>::getRow(size_t row) {
    return &dat[row * rows];
}

template<class T>
inline void Matrix<T>::set(const size_t row, const size_t col, const T &val) {
    dat[cols * row + col] = val;
}

template<class T>
inline T& Matrix<T>::get(const size_t row, const size_t col) {
    return dat[cols * row + col];
}

template<class T>
void Matrix<T>::randomize_rows() {
    for (size_t i = 0; i < rows; i++) {
        size_t row = (rand() / (double) RAND_MAX) * rows;
        if (i != row) {
            for (size_t j = 0; j < cols; j++)
                std::swap(get(i, j), get(row, j));
        }
    }
}

template<class T>
void Matrix<T>::randomize() {
    randomize(1);
}

template<class T>
void Matrix<T>::randomize(double normalize) {
    srand(time(NULL));
    for (size_t i = 0; i < rows * cols; i++) {
        dat[i] = (rand() / (double) RAND_MAX - .5) * 2.0 / normalize;
    }
}

template<class T>
Matrix<T>::~Matrix() {
    delete[] dat;
}

template<class K>
std::ostream& operator<<(std::ostream &cout, Matrix<K> const &m) {
    for (size_t i = 0; i < m.rows * m.cols; i++) {
        if (i % m.cols == 0 && i != 0)
            cout << std::endl;
        cout << m.dat[i] << " ";
    }
    return cout;
}

// allows compiler to link successfully
template std::ostream& operator<<(std::ostream &cout, Matrix<double> const &m);
template std::ostream& operator<<(std::ostream &cout, Matrix<float> const &m);
template std::ostream& operator<<(std::ostream &cout, Matrix<int> const &m);
template class Matrix<double>;
template class Matrix<float>;
template class Matrix<int>;
