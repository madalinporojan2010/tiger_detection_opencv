#ifndef OPENCVAPP_H
#define OPENCVAPP_H

namespace tiger_detection {
	void showHistogram(const std::string& name, std::vector<int> hist, const int  hist_cols, const int hist_height);

	std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> createPatches(Mat src, Mat src_hue, int patchSize);

	void showClusters(int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other));

	void showBinnedHistogram(int numberOfBins);

	void showImageFeatures();

	cv::Mat_<Vec3b> computeTigerImageClusters(const cv::String fname, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other));

	std::vector<std::thread> threadedTestImages(int testNumber, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other), std::string heuristicFuncName);

	void randomizedTesting(int numberOfTests);
};


#endif