//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#include "DataSet.hpp"
#include <sstream>
#include <vector>
#include <time.h>

#include <assert.h>
//#define NDEBUG

using namespace std;

template<class T>
DataSet<T>::DataSet(const size_t cols) :
        rows(0), cols(cols) {
    assert(cols>0);
}

template<class T>
DataSet<T>::DataSet(const size_t rows, const size_t cols) :
        dat(vector<T>(rows * cols)), rows(rows), cols(cols) {
    assert(cols>0 && rows>=0);
}

template<class T>
DataSet<T>::DataSet(const DataSet<T> &m) :
        dat(m.dat), rows(m.rows), cols(m.cols) {
}

template<class T>
void DataSet<T>::printRow(size_t row) {
    assert(row>=0 && row < rows);
    T* rowData = getRow(row);
    for (size_t i = 0; i < rows; i++)
        std::cout << rowData[i] << " ";
}

template<class T>
string DataSet<T>::sPrintRow(size_t row) {
    assert(row>=0 && row < rows);
    T* rowData = getRow(row);
    std::stringstream ss;
    for (size_t i = 0; i < rows; i++)
        ss << rowData[i] << " ";
    return ss.str();
}

template<class T>
void DataSet<T>::addRows(std::vector<T> row) {
    if (row.size() % cols != 0)
        throw "Row length != cols";
    for (size_t i = 0; i < row.size(); i++)
        dat.push_back(row[i]);
    rows += row.size() / cols;
}

template<class T>
void DataSet<T>::addRow(T* row) {
    for (size_t i = 0; i < cols; i++)
        dat.push_back(row[i]);
    rows++;
}

template<class T>
void DataSet<T>::addRows(T* row, size_t num) {
    for (size_t i = 0; i < cols * num; i++)
        dat.push_back(row[i]);
    rows += num;
}

template<class T>
inline T* DataSet<T>::getRow(size_t row) {
    assert(row>=0 && row < rows);
    return &dat[row * cols];
}

template<class T>
inline void DataSet<T>::set(const size_t row, const size_t col, const T &val) {
    assert(row>=0 && row < rows);
    assert(col>=0 && col < cols);
    dat[cols * row + col] = val;
}

template<class T>
inline T& DataSet<T>::get(const size_t row, const size_t col) {
    assert(row>=0 && row < rows);
    assert(col>=0 && col < cols);

    return dat[cols * row + col];
}

template<class T>
void DataSet<T>::randomize_rows() {
    srand(time(NULL));
    for (size_t i = 0; i < rows; i++) {
        size_t row = (rand() / (double) RAND_MAX) * rows;
        if (i != row) {
            for (size_t j = 0; j < cols; j++)
                std::swap(get(i, j), get(row, j));
        }
    }
}

template<class T>
void DataSet<T>::randomize_rows(DataSet<T> &m1, DataSet<T> &m2) {
    if (m1.rows != m2.rows)
        throw "rows(m1) != rows(m2)";
    srand(time(NULL));
    for (size_t i = 0; i < m1.rows; i++) {
        size_t row = (rand() / (double) RAND_MAX) * m1.rows;
        if (i != row) {
            for (size_t j = 0; j < m1.cols; j++)
                std::swap(m1.get(i, j), m1.get(row, j));
            for (size_t j = 0; j < m2.cols; j++)
                std::swap(m2.get(i, j), m2.get(row, j));

        }
    }
}

template<class T>
void DataSet<T>::randomize(double normalize = 1) {
    srand(time(NULL));
    for (size_t i = 0; i < rows * cols; i++) {
        dat[i] = (rand() / (double) RAND_MAX - .5) * 2.0 / normalize;
    }
}

template<class T>
DataSet<T>::~DataSet() {
}

template<class K>
std::ostream& operator<<(std::ostream &cout, DataSet<K> const &m) {
    for (size_t i = 0; i < m.rows * m.cols; i++) {
        if (i % m.cols == 0 && i != 0)
            cout << std::endl;
        cout << m.dat[i] << " ";
    }
    return cout;
}

template<class T>
template<class K> // be wary of implicit casts
bool DataSet<T>::equals(DataSet<K> &other) {
    if (other.rows != rows || other.cols != cols)
        return false;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
        if (get(i,j) != other.get(i,j))
            return false;
        }
    }
    return true;
}

template<class T>
DataSet<T>& DataSet<T>::operator=(const DataSet<T> &rhs) {
    this->dat = rhs.dat;
    this->cols = rhs.cols;
    this->rows = rhs.rows;
    return *this;
}

// allows compiler to link successfully for common primitive data types
template std::ostream& operator<<(std::ostream &cout, DataSet<double> const &m);
template std::ostream& operator<<(std::ostream &cout, DataSet<float> const &m);
template std::ostream& operator<<(std::ostream &cout, DataSet<int> const &m);
template std::ostream& operator<<(std::ostream &cout, DataSet<char> const &m);
// for = assignment
template DataSet<double>& DataSet<double>::operator=(const DataSet<double> &rhs);
template DataSet<float>& DataSet<float>::operator=(const DataSet<float> &rhs);
template DataSet<int>& DataSet<int>::operator=(const DataSet<int> &rhs);
template DataSet<char>& DataSet<char>::operator=(const DataSet<char> &rhs);
// allows casting between char/double
// for equals comparison (discrete classifications)
template bool DataSet<char>::equals(DataSet<double> &other);
template bool DataSet<double>::equals(DataSet<char> &other);
template bool DataSet<double>::equals(DataSet<double> &other);
template class DataSet<double> ;
template class DataSet<float> ;
template class DataSet<int> ;
template class DataSet<char> ;
