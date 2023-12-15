#ifndef OPENCVAPP_H
#define OPENCVAPP_H

namespace Segmentation {
	void showHistogram(const std::string& name, std::vector<int> hist, const int  hist_cols, const int hist_height);

	std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> createPatches(const std::string readType,Mat src_color, Mat src, Mat src_hue, int patchSize);

	void showClusters(int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other));

	void showBinnedHistogram(int numberOfBins);

	cv::Mat_<Vec3b> computeImageClusters(const cv::String fname, const std::string readType, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other));

	std::vector<std::thread> threadedTestImages(int testNumber, const std::string readType, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other), std::string heuristicFuncName);

	void randomizedTesting(int numberOfTests, const std::string readType);
};


#endif