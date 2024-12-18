#include "NeuralNetActivation.hpp"
#include <cmath>

void sigmoid(std::vector<double> &x)
{
    for (double &element : x)
    {
        element = 1.f / (1 + exp(-element));
    }
}

double sigmoidElement(const double x)
{
    return 1.f / (1 + exp(-x));
}

double sigmoidElementPrime(const double x)
{
    const double expNeg = exp(-x);
    return expNeg / ((expNeg + 1) * (expNeg + 1));
}


void sigmoidPrime(std::vector<double> &x)
{
    for (double &element : x)
    {
        const double expNeg = exp(-element);
        element = expNeg / ((expNeg + 1) * (expNeg + 1));
    }
}

void softmax(std::vector<double> &x)
{
    std::vector<double> expVect;
    double summedExp = 0;
    for (double element : x)
    {
        expVect.push_back(exp(element));
        summedExp += expVect.back();
    }
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = expVect[i] / summedExp;
    }
}

void softmax(Matrix2D &matrix)
{
    matrix.map([](const double x) -> double { return x * x; });
    std::vector<double>& elements = matrix.getElements();
    const size_t length = matrix.getLength();
    double* expVect = new double[length];
    //std::vector<double> expVect;
    double summedExp = 0;
    for (size_t i = 0; i < length; ++i)
    {
        expVect[i] = exp(elements[i]);
        summedExp += expVect[i];
    }
    for (size_t i = 0; i < length; ++i)
    {
        elements[i] = expVect[i] / summedExp;
    }

    delete[] expVect;
}

Matrix2D sigmoidElement(const Matrix2D& matrix)
{
    Matrix2D outputMatrix(matrix);
    outputMatrix.map(sigmoidElement);
    return outputMatrix;
}

Matrix2D sigmoidElementPrime(const Matrix2D& matrix)
{
    Matrix2D outputMatrix(matrix);
    outputMatrix = matrix;
    outputMatrix.map(sigmoidElementPrime);
    return outputMatrix;
}