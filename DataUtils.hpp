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
#include <vector>

template<class T = double>
class DataSet {
protected:
    std::vector<T> dat;
public:
    std::size_t rows; //TODO: add getter/setter for this and dat
    // columns fixed -- no partial data allowed (yet)
    const std::size_t cols;

    explicit DataSet(const std::size_t cols);
    explicit DataSet(const std::size_t rows, const std::size_t cols);
    explicit DataSet(const DataSet<T> &m);
    ~DataSet();
    void printRow(std::size_t row);
    std::string sPrintRow(size_t row);
    void addRows(std::vector<T> row);
    void addRows(T* row,size_t num);
    void addRow(T* row);
    T* getRow(std::size_t row);
    T& get(const std::size_t row, const std::size_t col);
    void set(const std::size_t row, const std::size_t col, const T &val);
    void randomize_rows();
    void randomize(double normalize);
    static void randomize_rows(DataSet<T> &m1, DataSet<T> &m2);
    template<class K> friend std::ostream & operator <<(std::ostream & cout, const DataSet<K> &m);
};

#endif /* DATAUTILS_HPP_ */
