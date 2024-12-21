#include "NeuralNetBuilder.hpp"
#include <iostream>
#include <random>

NeuralNetBuilder::NeuralNetBuilder()
    : numPreviousLayerNodes(0)
    , inputSet(false)
    , outputSet(false)
    , neuralNet()
{
}

NeuralNetBuilder::~NeuralNetBuilder()
{
}

NeuralNetBuilder& NeuralNetBuilder::setLearningRate(const double learningRate)
{
    neuralNet.learningRate = learningRate;
    return *this;
}

NeuralNetBuilder& NeuralNetBuilder::setMomentum(const double momentum)
{
    neuralNet.momentum = momentum;
    return *this;
}

// options => sigmoid
NeuralNetBuilder& NeuralNetBuilder::setHiddenLayerActivation(const std::string &functionName)
{
    if ("sigmoid" == functionName)
    {
        neuralNet.hiddenLayerActivationFunction = sigmoid;
        neuralNet.hiddenLayerActivationFunctionPrime = sigmoidPrime;
    }
    return *this;
}

// options => softmax
NeuralNetBuilder& NeuralNetBuilder::setOutputLayerActivation(const std::string &functionName)
{
    if ("softmax" == functionName)
    {
        neuralNet.outputLayerActivationFunction = softmax;
    }
    return *this;
}

NeuralNetBuilder& NeuralNetBuilder::setInput(const size_t numInputNodes)
{
    numPreviousLayerNodes = numInputNodes;    
    inputSet = true;
    //++neuralNet.numLayers;
    return *this;
}

NeuralNetBuilder& NeuralNetBuilder::setOutput(const size_t numOutputNodes)
{
    outputSet = true;
    addLayer(numOutputNodes);
    neuralNet.networkOutput = Matrix2D(numOutputNodes, 1);
    return *this;
}

NeuralNetBuilder& NeuralNetBuilder::addLayer(const size_t numNodes)
{
    // check this later
    neuralNet.inputs.emplace_back(numPreviousLayerNodes, 1);
    neuralNet.weights.emplace_back(numNodes, numPreviousLayerNodes);
    neuralNet.biases.emplace_back(numNodes, 1);
    neuralNet.zValues.emplace_back(numNodes, 1);
    neuralNet.activatedOutputs.emplace_back(numNodes, 1);

    ++neuralNet.numLayers;

    numPreviousLayerNodes = numNodes;
    return *this;
}

// randomize weights and biases
NeuralNetBuilder& NeuralNetBuilder::randomizeWeights()
{
    const double maxWeight = 0.01;
    const double minWeight = -0.01;
    std::random_device randDevice;
    std::uniform_real_distribution<double> uniformRealDistribution(minWeight, maxWeight);
    std::mt19937_64 randomEngine(randDevice());

    for (int x = 0; x < neuralNet.numLayers; ++x)
    {
        std::vector<double>& weights = neuralNet.weights[x].getElements();
        size_t length = neuralNet.weights[x].getLength();
        for (size_t i = 0; i < length; ++i)
        {
            weights[i] = uniformRealDistribution(randomEngine);
        }

        std::vector<double>& biases = neuralNet.biases[x].getElements();
        length = neuralNet.biases[x].getLength();
        for (size_t i = 0; i < length; ++i)
        {
            biases[i] = uniformRealDistribution(randomEngine);
        }
    }
    return *this;
}

NeuralNet& NeuralNetBuilder::build()
{
    if (!inputSet || !outputSet)
    {
        std::cout << "NeuralNetBuilder::build() Invalid Neural Net Built" << std::endl;
    }
    randomizeWeights();
    return neuralNet;
}


