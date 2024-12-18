#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>

struct TrainTest
{
    std::vector<std::vector<double>> xTrain;
    std::vector<int> yTrain;
    std::vector<std::vector<double>> xTest;
    std::vector<int> yTest;
};

//template<typename T>
class Dataset
{
public:
    std::vector<std::vector<double>> attributes;
    std::vector<int> predictor;
public:
    Dataset();

    void loadCsv(const std::string &filepath, const std::string &delimeter=",");
    void show();

    TrainTest trainTestSplit(const float trainRatio=0.8);

    void normalize(const size_t col);

    void normalize(const std::vector<size_t> columns);

    /**
      * Write dataset in csv format
      */
    void serialize(const std::string& filepath , const bool index = false);
};

#endif // DATASET_HPP