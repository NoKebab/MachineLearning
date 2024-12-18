#include <iostream>
#include <NeuralNet.hpp>
#include <NeuralNetBuilder.hpp>
#include <Dataset.hpp>
#include <chrono>
#include <Loss.hpp>

int main(void)
{
    std::cout << "FFNL APP" << std::endl;
    //Dataset trainDataset;
    //Dataset testDataset;
    //trainDataset.loadCsv("C:/Users/Peyton/dev/MachineLearning/datasets/BreastCancerTrainFold0.csv");
    //testDataset.loadCsv( "C:/Users/Peyton/dev/MachineLearning/datasets/BreastCancerTestFold0.csv");
    //soybeanDataset.show();
    Dataset dataset;
    dataset.loadCsv("C:/Users/Peyton/dev/MachineLearning/datasets/iris_processed.csv");
    TrainTest trainTest = dataset.trainTestSplit(0.8f);
    //trainTest.xTrain = irisDataset.attributes;
    //trainTest.yTrain = irisDataset.predictor;
    //trainTest.xTest = irisDataset.attributes;
    //trainTest.yTest = irisDataset.predictor;

    NeuralNetBuilder neuralNetBuilder;
    neuralNetBuilder.setLearningRate(0.5)
                    .setMomentum(0.1)
                    .setHiddenLayerActivation("sigmoid")
                    .setOutputLayerActivation("softmax")
                    .setInput(4)  // input  layer
                    .addLayer(5)  // hidden layer
                    .addLayer(5)  // hidden layer
                    .setOutput(3) // output layer
                    .randomizeWeights();
    NeuralNet neuralNet = neuralNetBuilder.build();

    neuralNet.serialize("C:/Users/Peyton/dev/MachineLearning/Temp/NetworkWeightsUntrained.txt");

    std::chrono::high_resolution_clock timingClock;
    auto startTime = timingClock.now();

    neuralNet.train(trainTest.xTrain, trainTest.yTrain);

    auto stopTime = timingClock.now();
    auto trainingTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
    std::cout << "Training time " << trainingTimeMs.count() << "ms" << std::endl;

    neuralNet.serialize("C:/Users/Peyton/dev/MachineLearning/Temp/NetworkWeightsTrained.txt");

    startTime = timingClock.now();
    std::vector<int>& predictions = neuralNet.test(trainTest.xTest);
    stopTime = timingClock.now();
    auto testingTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
    std::cout << "Testing time " << testingTimeMs.count() << "ms" << std::endl;

    Loss loss(trainTest.yTest, predictions, 3);
    printf("Accuracy %.2f%%\n", loss.getAccuracy() * 100);
    printf("F1 Macro %.2f%%\n", loss.getF1ScoreMacro() * 100);
    std::cout << "Confusion Matrix" << std::endl;
    loss.getConfusionMatrix().show();
}