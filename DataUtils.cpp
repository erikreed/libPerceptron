//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#include "DataUtils.hpp"
#include <sstream>
#include <vector>

using namespace std;

template<class T>
DataSet<T>::DataSet(const size_t cols) :
        rows(0), cols(cols) {
}

template<class T>
DataSet<T>::DataSet(const size_t rows, const size_t cols) :
        dat(vector<T>(rows * cols)), rows(rows), cols(cols) {
}

template<class T>
DataSet<T>::DataSet(const DataSet<T> &m) :
        dat(m.dat), rows(m.rows), cols(m.cols) {
}

template<class T>
void DataSet<T>::printRow(size_t row) {
    T* rowData = getRow(row);
    for (size_t i = 0; i < rows; i++)
        std::cout << rowData[i] << " ";
}

template<class T>
string DataSet<T>::sPrintRow(size_t row) {
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
    rows += row.size()/cols;
}

template<class T>
void DataSet<T>::addRow(T* row) {
    for (size_t i = 0; i < cols; i++)
        dat.push_back(row[i]);
    rows++;
}

template<class T>
void DataSet<T>::addRows(T* row, size_t num) {
    for (size_t i = 0; i < cols*num; i++)
        dat.push_back(row[i]);
    rows += num;
}

template<class T>
inline T* DataSet<T>::getRow(size_t row) {
    return &dat[row * rows];
}

template<class T>
inline void DataSet<T>::set(const size_t row, const size_t col, const T &val) {
    dat[cols * row + col] = val;
}

template<class T>
inline T& DataSet<T>::get(const size_t row, const size_t col) {
    return dat[cols * row + col];
}

template<class T>
void DataSet<T>::randomize_rows() {
    for (size_t i = 0; i < rows; i++) {
        size_t row = (rand() / (double) RAND_MAX) * rows;
        if (i != row) {
            for (size_t j = 0; j < cols; j++)
                std::swap(get(i, j), get(row, j));
        }
    }
}

template<class T>
void DataSet<T>::randomize() {
    randomize(1);
}

template<class T>
void DataSet<T>::randomize(double normalize) {
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

// allows compiler to link successfully
template std::ostream& operator<<(std::ostream &cout, DataSet<double> const &m);
template std::ostream& operator<<(std::ostream &cout, DataSet<float> const &m);
template std::ostream& operator<<(std::ostream &cout, DataSet<int> const &m);
template class DataSet<double> ;
template class DataSet<float> ;
template class DataSet<int> ;
