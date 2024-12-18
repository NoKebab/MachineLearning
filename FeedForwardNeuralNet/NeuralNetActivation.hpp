#ifndef NEURAL_NET_ACTIVATION_HPP
#define NEURAL_NET_ACTIVATION_HPP

#include <vector>
#include <Matrix2D.hpp>

typedef void (*activationFunction)(std::vector<double>&);

void sigmoid(std::vector<double> &x);

void sigmoidPrime(std::vector<double> &x);

void softmax(std::vector<double> &x);

void softmax(Matrix2D &matrix);

double sigmoidElement(const double x);

double sigmoidElementPrime(const double x);

Matrix2D sigmoidElement(const Matrix2D& matrix);

Matrix2D sigmoidElementPrime(const Matrix2D& matrix);

#endif // NEURAL_NET_ACTIVATION_HPP
 