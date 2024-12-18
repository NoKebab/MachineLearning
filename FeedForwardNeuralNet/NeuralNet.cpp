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
    const size_t epochs = 1000;
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
    // # get error of the last layer
    // last_layer_error = self.__last_layer_error(y_train_example)
    // # use recurrence relation to get the errors of all other layers
    // all_layer_errors = self.__layer_error(last_layer_error)
    // # calculate new changes in weights
    // changes_in_weights = self.__change_in_weights(all_layer_errors)
    // # update the weights for every layer
    // self.__update_weights(changes_in_weights)

    //    layer_num = self.num_hidden_layers - 1
    //    layer_errors = [last_layer_error]
    //    while layer_num >= 0:
    //        l_inputs = self.layers[layer_num]['inputs']
    //        gradient = self.hidden_activation_prime(l_inputs)
    //        next_layer = layer_num + 1
    //        next_layer_error = layer_errors[0]
    //        next_weights = self.layers[next_layer]['weights']
    //        this_layer_err = np.matmul(next_layer_error.reshape((1, next_layer_error.shape[0])), next_weights)[0]
    //        this_layer_err *= gradient
    //        layer_errors.insert(0, this_layer_err)
    //        layer_num -= 1
    //    return layer_errors

    //    m = X.shape[1]  # Number of examples
    //    grad_W = [None] * (self.num_layers - 1)
    //    grad_b = [None] * (self.num_layers - 1)

    //    # Compute output layer error
    //    delta = mse_loss_derivative(activations[-1], Y) * sigmoid_derivative(z_values[-1])
    //    grad_W[-1] = np.dot(delta, activations[-2].T) / m
    //    grad_b[-1] = np.sum(delta, axis=1, keepdims=True) / m

    //    # Backpropagate through hidden layers
    //    for l in range(self.num_layers - 2, 0, -1):
    //        delta = np.dot(self.weights[l].T, delta) * sigmoid_derivative(z_values[l - 1])
    //        grad_W[l - 1] = np.dot(delta, activations[l - 1].T) / m
    //        grad_b[l - 1] = np.sum(delta, axis=1, keepdims=True) / m

    //    return grad_W, grad_b

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