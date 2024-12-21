#include <iostream>
#include <NeuralNet.hpp>
#include <NeuralNetBuilder.hpp>
#include <Dataset.hpp>
#include <chrono>
#include <Loss.hpp>

struct DatasetModelCombo
{
    Dataset dataset;
    NeuralNet neuralNet;
};

DatasetModelCombo buildIrisNetwork()
{
    DatasetModelCombo dmc;
    dmc.dataset.loadCsv("C:/Users/Peyton/dev/MachineLearning/datasets/iris_preprocessed.csv");

    NeuralNetBuilder neuralNetBuilder;
    neuralNetBuilder.setLearningRate(0.1)
        .setMomentum(0.1)
        .setHiddenLayerActivation("sigmoid")
        .setOutputLayerActivation("softmax")
        .setInput(4)  // input  layer
        .addLayer(4)  // hidden layer
        .setOutput(3); // output layer
    dmc.neuralNet = neuralNetBuilder.build();
    return dmc;
}

DatasetModelCombo buildBcNetwork()
{
    DatasetModelCombo dmc;
    dmc.dataset.loadCsv("C:/Users/Peyton/dev/MachineLearning/datasets/BREAST_CANCER_WISCONSIN_preprocessed.csv");

    NeuralNetBuilder neuralNetBuilder;
    neuralNetBuilder.setLearningRate(0.2)
        .setMomentum(0.1)
        .setHiddenLayerActivation("sigmoid")
        .setOutputLayerActivation("softmax")
        .setInput(9)  // input  layer
        //.addLayer(5)  // hidden layer
        .setOutput(2); // output layer
    dmc.neuralNet = neuralNetBuilder.build();
    return dmc;
}

DatasetModelCombo buildGlassNetwork()
{
    DatasetModelCombo dmc;
    dmc.dataset.loadCsv("C:/Users/Peyton/dev/MachineLearning/datasets/GLASS_preprocessed.csv");

    NeuralNetBuilder neuralNetBuilder;
    neuralNetBuilder.setLearningRate(0.1)
        .setMomentum(0.1)
        .setHiddenLayerActivation("sigmoid")
        .setOutputLayerActivation("softmax")
        .setInput(9)  // input  layer
        .addLayer(5)  // hidden layer
        .setOutput(2); // output layer
    dmc.neuralNet = neuralNetBuilder.build();
    return dmc;
}

DatasetModelCombo buildXorNetwork()
{
    DatasetModelCombo dmc;
    dmc.dataset.loadCsv("C:/Users/Peyton/dev/MachineLearning/datasets/xor_data.csv");

    NeuralNetBuilder neuralNetBuilder;
    neuralNetBuilder.setLearningRate(0.2)
        .setMomentum(0.1)
        .setHiddenLayerActivation("sigmoid")
        .setOutputLayerActivation("softmax")
        .setInput(2)  // input  layer
        .addLayer(2)  // hidden layer
        .addLayer(2)  // hidden layer
        .setOutput(2); // output layer
    dmc.neuralNet = neuralNetBuilder.build();
    return dmc;
}

int main(void)
{
    std::cout << "FFNL APP" << std::endl;

    DatasetModelCombo allDmc[] = { buildIrisNetwork(), buildBcNetwork(), buildGlassNetwork(), buildXorNetwork() };
    std::string datasetNames[] = { "Iris", "BreastCancer", "Glass", "Xor" };

    const size_t datasetIndex = 1;

    std::string datasetName = datasetNames[datasetIndex];

    DatasetModelCombo dmc = allDmc[datasetIndex];
    TrainTest trainTest = dmc.dataset.trainTestSplit(0.8f);
    NeuralNet neuralNet = dmc.neuralNet;

    std::string untrainedWeightFileName = "C:/Users/Peyton/dev/MachineLearning/Temp/NetworkWeightsUntrained_";
    untrainedWeightFileName += datasetName;
    untrainedWeightFileName += ".txt";
    neuralNet.serialize(untrainedWeightFileName);

    std::chrono::high_resolution_clock timingClock;
    auto startTime = timingClock.now();

    neuralNet.train(trainTest.xTrain, trainTest.yTrain);

    auto stopTime = timingClock.now();
    auto trainingTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
    std::cout << "Training time " << trainingTimeMs.count() << "ms" << std::endl;

    std::string trainedWeightFileName = "C:/Users/Peyton/dev/MachineLearning/Temp/NetworkWeightsTrained_";
    trainedWeightFileName += datasetName;
    trainedWeightFileName += ".txt";
    neuralNet.serialize(trainedWeightFileName);

    startTime = timingClock.now();
    std::vector<int> predictions = neuralNet.test(trainTest.xTest);
    stopTime = timingClock.now();
    auto testingTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
    std::cout << "Testing time " << testingTimeMs.count() << "ms" << std::endl;

    Loss loss(trainTest.yTest, predictions);
    printf("Accuracy %.2f%%\n", loss.getAccuracy() * 100);
    printf("F1 Macro %.2f%%\n", loss.getF1ScoreMacro() * 100);
    std::cout << "Confusion Matrix" << std::endl;
    loss.getConfusionMatrix().show();
}