//============================================================================
// Name        :
// Author      : Erik Reed
// Description :
//============================================================================

#include "OptMLP.hpp"
#include <typeinfo>
#include "math.h"

using namespace std;

inline double round(double d) {
  return floor(d + 0.5);
}

void testOR(NeuralNetwork &NN) {
  cout << typeid(NN).name() << ": testOR" << endl;
  DataSet<double> test1(2);
  DataSet<double> test1out(1);
  double test1new[] = { 0, 0, 1, 0, 0, 1, 1, 1 };
  double test2new[] = { 0, 1, 1, 1 };
  test1.addRows(test1new, 4);
  test1out.addRows(test2new, 4);

  // training data = testing data
  NN.train(test1, test1out);
  double acc = NN.test(test1, test1out);
  if (acc != 100) {
    cerr << "testOR training failed" << endl;
    throw 1;
  }
  DataSet<double>* eval = NN.evaluate(test1);
  if (!eval->equals(test1out))
    throw "testOR eval failed";
  delete eval;
  cout << endl;
}

void testAND(NeuralNetwork &NN) {
  cout << typeid(NN).name() << ": testAND" << endl;
  DataSet<double> test1(2);
  DataSet<double> test1out(1);
  double test1new[] = { 0, 0, 1, 0, 0, 1, 1, 1 };
  double test2new[] = { 0, 0, 0, 1 };
  test1.addRows(test1new, 4);
  test1out.addRows(test2new, 4);

  // training data = testing data
  NN.train(test1, test1out);
  double acc = NN.test(test1, test1out);
  if (acc != 100) {
    cerr << "testXOR training failed" << endl;
    throw 1;
  }
  DataSet<double>* eval = NN.evaluate(test1);
  if (!eval->equals(test1out)) {
    cerr << "testAND eval failed" << endl;
    throw 1;
  }
  delete eval;
  cout << endl;
}

void testXOR(NeuralNetwork &NN) {
  cout << typeid(NN).name() << ": testXOR" << endl;
  DataSet<double> test1(2);
  DataSet<double> test1out(1);
  double test1new[] = { 0, 0, 1, 0, 0, 1, 1, 1 };
  double test2new[] = { 0, 1, 1, 0 };
  test1.addRows(test1new, 4);
  test1out.addRows(test2new, 4);

  // training data = testing data
  NN.train(test1, test1out);
  double acc = NN.test(test1, test1out);
  if (acc != 100) {
    throw "testXOR training failed";
  }
  DataSet<double>* eval = NN.evaluate(test1);
  if (!eval->equals(test1out)) {
    cerr << "testXOR eval failed" << endl;
    throw 1;
  }
  delete eval;
  cout << endl;

}

void testPerceptron() {
  cout << "--- Testing Perceptron ---" << endl;
  Perceptron p1, p2;
  testOR(p1);
  testAND(p2);
  MLPerceptron mlp1, mlp2, mlp3;
  testOR(mlp1);
  testAND(mlp2);
  testXOR(mlp3);
}

int main(int argc, char** args) {
  DataSet<char> d('data.csv');

  cout << d << endl;

  return 0;
}

