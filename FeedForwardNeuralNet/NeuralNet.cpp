#include "NeuralNet.hpp"
#include <iostream>
#include <fstream>
#include <limits>

NeuralNet::NeuralNet()
    : learningRate(1) 
    , momentum(1)
    , hiddenLayerActivationFunction(sigmoid) 
    , hiddenLayerActivationFunctionPrime(sigmoidPrime)
    , outputLayerActivationFunction(softmax)
    , numLayers(0)
{
}

void NeuralNet::train(std::vector<std::vector<double>> &xTrain, std::vector<int> &yTrain)
{
    if (xTrain.size() != yTrain.size())
    {
        std::cout << "NeuralNet::train() size mismatch between xTrain and yTrain " << xTrain.size() << " " << yTrain.size() << std::endl;
        return;
    }
    const size_t epochs = 100;
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        //feedForward(xTrain);
        //backPropagation(yTrain);
        for (size_t i = 0; i < xTrain.size(); ++i)
        {
            feedForward(xTrain[i]);
            backPropagation(yTrain[i]);
        }
    }
}

std::vector<int> NeuralNet::test(std::vector<std::vector<double>> &xTest)
{
    std::vector<int> predictions;
    for (std::vector<double> &featureVector : xTest)
    {
        feedForward(featureVector);

        // make prediction => choose highest element
        // argmax
        int prediction = 0;
        double maxPred = std::numeric_limits<double>().min();
        std::vector<double>& output = activatedOutputs.back().getElements();
        const size_t numOutputs = activatedOutputs.back().getLength();
        for (size_t i = 0; i < numOutputs; i++)
        {
            if (maxPred < output[i]) 
            {
                maxPred = output[i];
                prediction = static_cast<int>(i);
            }
        }
        predictions.push_back(prediction);
    }
    return predictions;
}

void NeuralNet::feedForward(std::vector<std::vector<double>> &featureData)
{

    // update z value, activated, and input size to reflect number of samples
    inputs[0] = Matrix2D(featureData);
    inputs[0] = *inputs[0].getTranspose();

    for (size_t i = 0; i < numLayers; ++i)
    {
        const bool lastLayer = i == (numLayers - 1);
        // update z value, activated, and input size to reflect number of samples
        zValues[i] = Matrix2D(weights[i].getRows(), inputs[i].getCols());
        activatedOutputs[i] = Matrix2D(weights[i].getRows(), inputs[i].getCols());

        Matrix2D::matrixMultiply(weights[i], inputs[i], zValues[i]);
        zValues[i].add(biases[i]);
        activatedOutputs[i] = zValues[i];
        activatedOutputs[i].map(sigmoidElement);
        if (!lastLayer)
        {
            inputs[i + 1] = activatedOutputs[i];
        }
    }
}

void NeuralNet::backPropagation(std::vector<int> &targets)
{
    const size_t numExamples = targets.size();
    // BEGIN create error and update matrices
    Matrix2D layerDelta; 
    std::vector<Matrix2D> weightDeltas(numLayers);
    std::vector<Matrix2D> biasDeltas(numLayers);
    // TODO: make construction of this static 
    for (size_t i = 0; i < numLayers; ++i)
    {
        weightDeltas[i] = Matrix2D(weights[i].getRows(), weights[i].getCols());
        biasDeltas[i] = Matrix2D(biases[i].getRows(), biases[i].getCols());
    }
    // END create error and update matrices
    
    // last layer delta
    Matrix2D trueLabels(activatedOutputs[numLayers - 1].getRows(), numExamples);
    for (size_t i = 0; i < numExamples; ++i)
    {
        trueLabels.setElement(targets[i], i, 1.f);

    }
	Matrix2D delta(activatedOutputs.back());
	delta.sub(trueLabels);
    
    // previous layers deltas
}

void NeuralNet::feedForward(std::vector<double> &featureVector)
{
    inputs[0] = Matrix2D(featureVector);

    for (size_t i = 0; i < numLayers; ++i)
    {
        const bool lastLayer = i == (numLayers - 1);
        //Matrix2D *output = lastLayer ? &networkOutput : &inputs[i + 1];
        //weights[i].show();
        //inputs[i].show();
        Matrix2D::matrixMultiply(weights[i], inputs[i], zValues[i]);
        zValues[i].add(biases[i]);
        activatedOutputs[i] = zValues[i];
        activatedOutputs[i].map(sigmoidElement);

        if (!lastLayer)
        {
            inputs[i + 1] = activatedOutputs[i];
        }
        //*output = zValues[i];
        //output->map(sigmoidElement);
        
        //output->map(sigmoidElement);
        //if (lastLayer)
        //{
        //    softmax(*output);
        //}
        //else
        //{
        //    output->map(sigmoidElement);
        //}
    }
}

void NeuralNet::backPropagation(const int target)
{
    // BEGIN create error and update matrices
    std::vector<Matrix2D> layerErrors(numLayers); 
    std::vector<Matrix2D> weightDeltas(numLayers);
    std::vector<Matrix2D> biasDeltas(numLayers);
    // TODO: make construction of this static 
    for (size_t i = 0; i < numLayers; ++i)
    {
        layerErrors[i] = Matrix2D(activatedOutputs[i].getRows(), activatedOutputs[i].getCols());
        weightDeltas[i] = Matrix2D(weights[i].getRows(), weights[i].getCols());
        biasDeltas[i] = Matrix2D(biases[i].getRows(), biases[i].getCols());
    }
    // END create error and update matrices

    // BEGIN last layer error
    // one hot encode true labels for classification
    Matrix2D trueLabels(activatedOutputs[numLayers - 1].getRows(), activatedOutputs[numLayers - 1].getCols());
    trueLabels.setElement(target, 0, 1.f);

    Matrix2D gradient(zValues[numLayers - 1]);
    gradient.map(sigmoidElementPrime);

    layerErrors[numLayers - 1] = activatedOutputs[numLayers - 1];
    layerErrors[numLayers - 1].sub(trueLabels);
    layerErrors[numLayers - 1].mul(gradient);
    // END last layer error

    // BEGIN last layer weight & bias updates
    Matrix2D::matrixMultiply(layerErrors[numLayers - 1], *inputs[numLayers - 1].getTranspose(), weightDeltas[numLayers - 1]);
    biasDeltas[numLayers - 1] = layerErrors[numLayers - 1];
    // END last layer weight & bias updates

    // BEGIN backpropagation
    for (int i = numLayers - 2; i >= 0; --i)
    {
        Matrix2D::matrixMultiply(*weights[i + 1].getTranspose(), layerErrors[i + 1], layerErrors[i]);
        gradient = Matrix2D(zValues[i]);
        gradient.map(sigmoidElementPrime);
        layerErrors[i].mul(gradient);

        Matrix2D::matrixMultiply(layerErrors[i], *inputs[i].getTranspose(), weightDeltas[i]);
        biasDeltas[i] = layerErrors[i];
    }
    // END backpropagation

    // update weights and biases
    for (size_t i = 0; i < numLayers; ++i)
    {
        weightDeltas[i].scale(learningRate);
        biasDeltas[i].scale(learningRate);
        weights[i].sub(weightDeltas[i]);
        biases[i].sub(biasDeltas[i]);
    }
}

void NeuralNet::serialize(const std::string &filepath)
{
    std::ofstream outputFile(filepath);
    if (!outputFile.is_open())
    {
        std::cout << "NeuralNet::serialize() Could not open file " << filepath << std::endl;
        return;
    }

    for (size_t i = 0; i < numLayers; ++i)
    {
        outputFile << "Layer " << i << std::endl;
        outputFile << "Inputs" << std::endl;
        outputFile << inputs[i].getStringRepresentation();
        outputFile << "Weights" << std::endl;
        outputFile << weights[i].getStringRepresentation();
        outputFile << "Biases" << std::endl;
        outputFile << biases[i].getStringRepresentation();
        outputFile << "Z Values" << std::endl;
        outputFile << zValues[i].getStringRepresentation();
        outputFile << "Activated Outputs" << std::endl;
        outputFile << activatedOutputs[i].getStringRepresentation();
        outputFile << std::endl;
    }
    //outputFile << "Outputs" << std::endl;
    //outputFile << networkOutput.getStringRepresentation();

    outputFile.close();
}