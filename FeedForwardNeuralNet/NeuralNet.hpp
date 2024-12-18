#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <string>
#include <Matrix2D.hpp>
#include "NeuralNetActivation.hpp"

class NeuralNet
{
protected:
    friend class NeuralNetBuilder;

    // hyperparameters
    double learningRate; 
    double momentum; 

    activationFunction hiddenLayerActivationFunction;    
    activationFunction hiddenLayerActivationFunctionPrime;
    activationFunction outputLayerActivationFunction;    

    size_t numLayers;

    std::vector<Matrix2D> inputs;
    std::vector<Matrix2D> weights;
    std::vector<Matrix2D> biases;
    std::vector<Matrix2D> zValues;
    std::vector<Matrix2D> activatedOutputs;
    Matrix2D networkOutput;

private:
    void feedForward(std::vector<std::vector<double>> &featureData);
    void feedForward(std::vector<double> &featureVector);
    void backPropagation(std::vector<int> &targets);
    void backPropagation(const int target);
public:
    NeuralNet();

    void train(std::vector<std::vector<double>> &xTrain, std::vector<int> &yTrain);
    std::vector<int> test(std::vector<std::vector<double>> &xTest);

    void serialize(const std::string &filepath);
};

#endif // NEURAL_NET_HPP