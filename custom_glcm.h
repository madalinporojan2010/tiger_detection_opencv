#ifndef CUSTOM_GLCM_H
#define CUSTOM_GLCM_H

#define EPS 0.00000001

namespace custom_glcm {
	double Entropy(std::vector<double> vec);

	void meanStd(std::vector<double> v, double& m, double& stdev);

	std::vector<double> MargProbx(cv::Mat cooc);

	std::vector<double> MargProby(cv::Mat cooc);

	std::vector<double> ProbSum(cv::Mat cooc);

	std::vector<double> ProbDiff(cv::Mat cooc);

	double HaralickEnergy(cv::Mat cooc);

	double HaralickEntropy(cv::Mat cooc);

	double HaralickInverseDifference(cv::Mat cooc);

	double HaralickCorrelation(cv::Mat cooc, std::vector<double> probx, std::vector<double> proby);

	double HaralickInfoMeasure1(cv::Mat cooc, double ent, std::vector<double> probx, std::vector<double> proby);

	double HaralickInfoMeasure2(cv::Mat cooc, double ent, std::vector<double> probx, std::vector<double> proby);

	double HaralickContrast(cv::Mat cooc, std::vector<double> diff);

	double HaralickDiffEntropy(cv::Mat cooc, std::vector<double> diff);

	double HaralickDiffVariance(cv::Mat cooc, std::vector<double> diff);

	double HaralickSumAverage(cv::Mat cooc, std::vector<double> sumprob);

	double HaralickSumEntropy(cv::Mat cooc, std::vector<double> sumprob);

	double HaralickSumVariance(cv::Mat cooc, std::vector<double> sumprob);

	cv::Mat MatCooc(cv::Mat img, int N, int deltax, int deltay);

	cv::Mat MatCoocAdd(cv::Mat img, int N, std::vector<int> deltax, std::vector<int> deltay);

	std::vector<double> getFeatures(cv::Mat_<uchar> img);

};


#endif