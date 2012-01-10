//============================================================================
// Name        : Opt-MLP-lib
// Author      : Erik Reed
// Description : Implementation of an optimized, multithreaded,
//               single and multi-layer perceptron library in C++.
//               Fast neural network.
//============================================================================

#ifndef OPTMLP_HPP_
#define OPTMLP_HPP_

// comment out to disable excessive debugging message
#define DEBUG

#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif

#include "DataUtils.hpp"
#include "NeuralNetwork.hpp"

// typical 0.1 < ETA < 0.4
const double ETA = .2; // weight coefficient
const size_t MAX_ITERATIONS = 15;

#endif /* OPTMLP_HPP_ */
