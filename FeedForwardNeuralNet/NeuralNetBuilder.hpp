#ifndef NEUERAL_NET_BUILDER_HPP
#define NEUERAL_NET_BUILDER_HPP

#include "NeuralNet.hpp"
#include "NeuralNetActivation.hpp"
#include <string>
#include <memory>

class NeuralNetBuilder 
{
private:
    NeuralNet neuralNet;
    size_t numPreviousLayerNodes;

    // minimum non default requirements for working network
    bool inputSet;
    bool outputSet;
public:
    NeuralNetBuilder();
    ~NeuralNetBuilder();

    NeuralNetBuilder& setLearningRate(const double learningRate);
    NeuralNetBuilder& setMomentum(const double momentum);
    // options => sigmoid
    NeuralNetBuilder& setHiddenLayerActivation(const std::string &functionName);
    // options => softmax
    NeuralNetBuilder& setOutputLayerActivation(const std::string &functionName);
    NeuralNetBuilder& setInput(const size_t numInputNodes);
    NeuralNetBuilder& addLayer(const size_t numNodes);
    NeuralNetBuilder& setOutput(const size_t numOutputNodes);
    NeuralNetBuilder& randomizeWeights();

    NeuralNet& build();
};


#endif // NEUERAL_NET_BUILDER_HPP
