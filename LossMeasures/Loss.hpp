#ifndef MACHINE_LEARNING_LOSS_HPP
#define MACHINE_LEARNING_LOSS_HPP

#include <Matrix2D.hpp>

class Loss
{
private:
	Matrix2D confusionMatrix;
public:
	Loss(const std::vector<int>& truth, const std::vector<int>& predictions, const size_t numClasses);

	int getTruePositives(const int classNumber) const;
	int getFalsePositives(const int classNumber) const;
	int getFalseNegatives(const int classNumber) const;

	double getPrecision(const int classNumber) const;
	double getRecall(const int classNumber) const;
	double getF1Score(const int classNumber) const;

	double getPrecisionMacro() const;
	double getRecallMacro() const;
	double getF1ScoreMacro() const;

	double getAccuracy() const;

	Matrix2D getConfusionMatrix();
};

#endif // MACHINE_LEARNING_LOSS_HPP