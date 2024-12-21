#include "Dataset.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <array>
#include <random>
#include <map>


Dataset::Dataset()
{

}

/**
 * @brief load a csv file with the last column as the predictor (y value)
 * 
 * @tparam T predictor type
 * @param filepath path to input file
 */
void Dataset::loadCsv(const std::string &filepath, const std::string &delimiter)
{
    std::ifstream inputFile(filepath);
    if (!inputFile.is_open())
    {
        std::cerr << "Dataset<T>::loadCsv() Unable to open file " << filepath << std::endl;
        return;
    }
    // parse csv
    std::string line;
    size_t lineNumber = 0;
    while (getline(inputFile, line))
    {
        std::vector<std::string> tokens;
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delimiter)) != std::string::npos) 
        {
            token = line.substr(0, pos);
            tokens.push_back(token);
            line.erase(0, pos + delimiter.length());
        }
        tokens.push_back(line);

        attributes.emplace_back();
        // load parsed string into data
        for (size_t i = 0; i < tokens.size() - 1; ++i)
        {
            attributes.back().push_back(std::stod(tokens[i]));
            //if (attributes.size() <= i)
            //{
            //    attributes.push_back(std::vector<double>());
            //}
            //attributes[i].push_back(std::stod(tokens[i]));
        }
        predictor.push_back(std::stoi(tokens.back()));
        ++lineNumber;
    }

    inputFile.close();
}

void Dataset::show()
{
    for (size_t i = 0; i < attributes.size(); ++i)
    {
        for (size_t j = 0; j <= attributes.front().size(); ++j)
        {
            if (j == attributes.front().size())
            {
                std::cout << predictor[i] << std::endl;
            }
            else
            {
                std::cout << attributes[i][j] << ",";
            }
        }
    }
}

TrainTest Dataset::trainTestSplit(const float trainRatio)
{
    TrainTest trainTest;
    if (!((trainRatio > 0) && (trainRatio < 1)))
    {
        std::cout << "Dataset::trainTestSplit() invalid ratio must be bounded (0, 1)" << std::endl;
    }
    const size_t numTrainingElements = predictor.size() * trainRatio;

    // grab a list of random indices to get an even sample of the dataset
    std::random_device rd;
    std::mt19937_64 g(rd());
    std::vector<size_t> indices;
    for (size_t i = 0; i < predictor.size(); ++i)
    {
        indices.push_back(i);
    }
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < predictor.size(); ++i)
    {
        size_t index = indices.back();
        indices.pop_back();
        if (i < numTrainingElements)
        {
            trainTest.xTrain.push_back(attributes[index]);
            trainTest.yTrain.push_back(predictor[index]);
        }
        else
        {
            trainTest.xTest.push_back(attributes[index]);
            trainTest.yTest.push_back(predictor[index]);
        }
    }
    return trainTest;
}

TrainTest Dataset::trainTestSplitEqualClasses(const float trainRatio)
{
    std::map<int, int> numClassOccurences;
    for (size_t i = 0; i < predictor.size(); ++i)
    {
        if (numClassOccurences.contains(predictor[i]))
        {
            numClassOccurences[predictor[i]] += 1;
        }
        else
        {
            numClassOccurences[predictor[i]] = 1;
        }
    }

    // find class with least representation 
    // then sample the same number of elements from every other class

    int minNumClasses = numClassOccurences[0];
    int minNumClassesIndex = 0;
    for (auto const& [classNumber, numOccurences] : numClassOccurences)
    {
        if (numOccurences < minNumClasses)
        {
            minNumClasses = numOccurences;
            minNumClassesIndex = classNumber;
        }
    }

    TrainTest trainTest;
    const size_t numTrainingElements = minNumClasses * trainRatio;
    const size_t numTestingElements = minNumClasses - numTrainingElements;

    for (size_t classIndex = 0; classIndex < numClassOccurences.size(); ++classIndex)
    {
        Dataset classDataset = getAllByClass(classIndex);

        // grab a list of random indices to get an even sample of the dataset
        std::random_device rd;
        std::mt19937_64 g(rd());
        std::vector<size_t> indices(classDataset.predictor.size());
        for (size_t i = 0; i < classDataset.predictor.size(); ++i)
        {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), g);

        for (size_t i = 0; i < (numTrainingElements + numTestingElements); ++i)
        {
            if (i < numTrainingElements)
            {
                trainTest.xTrain.emplace_back(classDataset.attributes[indices[i]]);
                trainTest.yTrain.emplace_back(classDataset.predictor[indices[i]]);
            }
            else
            {
                trainTest.xTest.emplace_back(classDataset.attributes[indices[i]]);
                trainTest.yTest.emplace_back(classDataset.predictor[indices[i]]);
            }
        }
    }

    return trainTest;
}

/**
  * normalize a column between 0 and 1
  */
void Dataset::normalize(const size_t col)
{
    std::vector<double> column;
    for (size_t row = 0; row < attributes.size(); ++row)
    {
        column.push_back(attributes[row][col]);
    }

    double minElement = column[0];
    double maxElement = column[0];
    for (const double& element : column)
    {
        if (element > maxElement)
        {
            maxElement = element;
        }
        if (element < minElement)
        {
            minElement = element;
        }
    }

    const double range = maxElement - minElement;
    for (size_t row = 0; row < attributes.size(); ++row)
    {
        attributes[row][col] = (column[row] - minElement) / range;
    }
    //for (double& element : attributes[col])
    //{
    //    element = (element - minElement) / range;
    //}
}

void Dataset::normalize(const std::vector<size_t> columns)
{
    for (const size_t col : columns)
    {
        normalize(col);
    }
}

void Dataset::serialize(const std::string& filepath, const bool index)
{
    std::ofstream outCsvFile(filepath);
    if (!outCsvFile.is_open())
    {
        std::cout << "Dataset::serialize() can not open " << filepath << std::endl;
        return;
    }

    for (size_t row = 0; row < attributes.size(); ++row)
    {
        if (index)
        {
            outCsvFile << index << ",";
        }
		for (size_t col = 0; col < attributes.front().size(); ++col)
		{
            outCsvFile << attributes[row][col] << ",";
		}
        outCsvFile << predictor[row] << std::endl;
    }

    outCsvFile.close();
}

Dataset Dataset::getAllByClass(const int classIndex)
{
    Dataset dataset;

    for (int i = 0; i < predictor.size(); ++i)
    {
        if (predictor[i] == classIndex)
        {
            dataset.attributes.emplace_back(attributes[i]);
            dataset.predictor.emplace_back(predictor[i]);
        }
    }

    return dataset;
}
