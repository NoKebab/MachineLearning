#include "Loss.hpp"
#include <map>

Loss::Loss(const std::vector<int>& truth, const std::vector<int>& predictions)
{
	// the number of unique elements determines the size of the confusion matrix
	std::map<int, int> numClassOccurences;
	for (size_t i = 0; i < predictions.size(); ++i)
	{
		if (numClassOccurences.contains(truth[i]))
		{
			numClassOccurences[truth[i]] += 1;
		}
		else
		{
			numClassOccurences[truth[i]] = 1;
		}
	}
	confusionMatrix = Matrix2D(numClassOccurences.size(), numClassOccurences.size());
	for (size_t i = 0; i < predictions.size(); ++i)
	{
		confusionMatrix.setElement(truth[i], predictions[i], confusionMatrix.getElement(truth[i], predictions[i]) + 1);
	}
}

int Loss::getTruePositives(const int classNumber) const
{
	return confusionMatrix.getElement(classNumber, classNumber);
}

int Loss::getFalsePositives(const int classNumber) const
{
	int sum1 = 0;
	for (size_t row = 0; row < confusionMatrix.getRows(); ++row)
	{
		sum1 += static_cast<int>(confusionMatrix.getElement(row, classNumber));
	}
	return sum1 - getTruePositives(classNumber);
}

int Loss::getFalseNegatives(const int classNumber) const
{
	int sum1 = 0;
	for (size_t col = 0; col < confusionMatrix.getCols(); ++col)
	{
		sum1 += static_cast<int>(confusionMatrix.getElement(classNumber, col));
	}
	return sum1 - getTruePositives(classNumber);
}

double Loss::getPrecision(const int classNumber) const
{
	const int truePositves = getTruePositives(classNumber);
	const int falsePositives = getFalsePositives(classNumber);
	return truePositves / static_cast<double>(truePositves + falsePositives);
}

double Loss::getRecall(const int classNumber) const
{
	const int truePositves = getTruePositives(classNumber);
	const int falseNegatives = getFalseNegatives(classNumber);
	return truePositves / static_cast<double>(truePositves + falseNegatives);
}

double Loss::getF1Score(const int classNumber) const
{
	const double precision = getPrecision(classNumber);
	const double recall = getRecall(classNumber);
	if (0 == (precision + recall))
	{
		return 0;
	}
	return 2 * ((precision * recall) / (precision + recall));
}

double Loss::getPrecisionMacro() const
{
	std::vector<double> allTruePositives = confusionMatrix.diagonal();
	std::vector<double> allFalsePositives;
	for (size_t i = 0; i < confusionMatrix.getRows(); ++i)
	{
		allFalsePositives.push_back(getFalsePositives(i));
	}
	double summed = 0;
	for (size_t i = 0; i < allTruePositives.size(); ++i)
	{
		if ((allTruePositives[i] != 0) || (allFalsePositives[i] != 0))
		{
			summed += allTruePositives[i] / (allTruePositives[i] + allFalsePositives[i]);
		}
	}
	return (1.f / confusionMatrix.getRows()) * summed;
}

double Loss::getRecallMacro() const
{
	std::vector<double> allTruePositives = confusionMatrix.diagonal();
	std::vector<double> allFalseNegatives;
	for (size_t i = 0; i < confusionMatrix.getRows(); ++i)
	{
		allFalseNegatives.push_back(getFalseNegatives(i));
	}
	double summed = 0;
	for (size_t i = 0; i < allTruePositives.size(); ++i)
	{
		if ((allTruePositives[i] != 0) || (allFalseNegatives[i] != 0))
		{
			summed += allTruePositives[i] / (allTruePositives[i] + allFalseNegatives[i]);
		}
	}
	return (1.f / confusionMatrix.getRows()) * summed;
}

double Loss::getF1ScoreMacro() const
{
	double precisionMacro = getPrecisionMacro();
	double recallMacro = getRecallMacro();
	if ((0 != precisionMacro) || (0 != recallMacro))
	{
		return 2 * ((precisionMacro * recallMacro) / (precisionMacro + recallMacro));
	}
	return 0.f;
}

double Loss::getAccuracy() const
{
	double summedMatrix = 0;
	for (size_t row = 0; row < confusionMatrix.getRows(); ++row)
	{
		for (size_t col = 0; col < confusionMatrix.getCols(); ++col)
		{
			summedMatrix += confusionMatrix.getElement(row, col);
		}
	}
	double truePositives = 0;
	for (int i = 0; i < confusionMatrix.getRows(); ++i)
	{
		truePositives += static_cast<double>(getTruePositives(i));
	}
	if (summedMatrix < 1)
	{
		return 0;
	}
	return truePositives / summedMatrix;
}

Matrix2D Loss::getConfusionMatrix()
{
	return confusionMatrix;
}
