// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "algorithms.h"
#include "custom_glcm.h"
#include "OpenCVApplication.h"
#include "file_tester.h"

void tiger_detection::showHistogram(const std::string& name, std::vector<int> hist, const int  hist_cols, const int hist_height) {
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> tiger_detection::createPatches(Mat src, Mat src_hue, int patchSize) {
	int height = src.rows;
	int width = src.cols;

	std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> patches;

	cv::Size patch_size(patchSize, patchSize); // Specify the patch size

	for (int i = 0; i < src.rows; i += patch_size.height) {
		for (int j = 0; j < src.cols; j += patch_size.width) {
			cv::Rect patch_rect(j, i, min(patch_size.width, width - j - 1), min(patch_size.height, height - i - 1));
			cv::Mat patch = src(patch_rect);

			std::vector<double> features = custom_glcm::getFeatures(patch);
			std::vector<int> histoPatch = algorithms::binnedHistogram(src_hue(patch_rect), patchSize);
			std::vector<int> histoPatchNorm;

			for (int i = 0; i < histoPatch.size(); i++) {
				histoPatchNorm.push_back(histoPatch.at(i) / (patchSize * patchSize));
			}

			features.insert(features.end(), histoPatch.begin(), histoPatch.end());

			patches.push_back(std::make_tuple(patch_rect, patch, features));
		}
	}
	return patches;
}

void tiger_detection::showClusters(int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other)) {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src_color = imread(fname);
		Mat_<Vec3b> dst = tiger_detection::computeTigerImageClusters(fname, iterations, Kclusters, patchSize, heuristicFunc);

		imshow("original image", src_color);
		imshow("clustered image", dst);

		waitKey();
	}
}

void tiger_detection::showBinnedHistogram(int numberOfBins) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		int x = 0, y = 0, height = 8, width = 8;

		std::cout << "Image height: " << src.rows << std::endl;
		std::cout << "Image width: " << src.cols << std::endl;

		std::cout << "Patch x:\n";
		std::cin >> x;
		std::cout << "Patch y:\n";
		std::cin >> y;
		std::cout << "Patch height:\n";
		std::cin >> height;
		std::cout << "Patch width:\n";
		std::cin >> width;


		cv::Rect patch_rect(max(y, 0), max(x, 0), min(width, src.cols - y - 1), min(height, src.rows - x - 1));

		std::vector<int> hist = algorithms::binnedHistogram(src(patch_rect), numberOfBins);

		std::vector<int> showedHist;

		for (int h : hist) {
			for (int i = 0; i < 256 / hist.size(); i++) {
				showedHist.push_back(h);
			}
		}

		imshow("Original Image", src);
		imshow("Image Patch", src(patch_rect));
		tiger_detection::showHistogram("Binned Histogram for given patch", showedHist, showedHist.size(), 100);
		waitKey();
	}
}

void tiger_detection::showImageFeatures() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		double t = (double)getTickCount(); // Get the current time [s]

		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);


		for (const auto feature : custom_glcm::getFeatures(src)) {

			std::cout << feature<<std::endl;
		}
		waitKey();
	}
}

cv::Mat_<Vec3b> tiger_detection::computeTigerImageClusters(const cv::String fname, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other)) {
	
	Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
	Mat_<Vec3b> src_color = imread(fname);
	Mat_<Vec3b> src_hsv = imread(fname);
	std::vector<Mat> hsv_planes;
	Mat_<uchar> src_hue;

	cv::cvtColor(src_color, src_hsv, COLOR_BGR2HSV);
	split(src_hsv, hsv_planes);
	src_hue = hsv_planes[0]; // hue channel


	int height = src.rows;
	int width = src.cols;
	if (patchSize > max(height, width) || patchSize <= 0) {
		patchSize = max(height, width);
	}

	Mat_<Vec3b> dst(height, width);
	dst.setTo(cv::Scalar(255, 255, 255));

	std::vector<algorithms::Point> points;


	std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> patches = tiger_detection::createPatches(src, src_hue, patchSize);

	for (const auto patch : patches) {
		double centerX = (double)std::get<0>(patch).width / 2.0;
		double centerY = (double)std::get<0>(patch).height / 2.0;

		algorithms::Point point = algorithms::Point(centerX, centerY, std::get<0>(patch));
		point.features = std::get<2>(patch);

		points.push_back(point);

	}

	std::vector<algorithms::Point> centroids;
	centroids = algorithms::kMeansClustering(&points, iterations, Kclusters, heuristicFunc);

	int markSize = 20;
	int markThickness = 2;
	cv::Scalar markColor = cv::Scalar(0, 0, 0);

	std::vector<int> clusterIds;
	std::transform(points.begin(), points.end(), std::back_inserter(clusterIds), [](algorithms::Point x) {
		return x.cluster;
		});

	int maxClusterId = *std::max_element(clusterIds.begin(), clusterIds.end());

	std::vector<Vec3b> randomColors = algorithms::getRandomColors(maxClusterId + 1);

	for (auto const& point : points) {
		dst(point.patchRect).setTo(cv::Scalar(randomColors[point.cluster]));
	}

	return dst;
}

/* Threaded testing.... */

void thFunction(int testNumber, std::string fname, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other)) {
	Mat_<Vec3b> dstImage = tiger_detection::computeTigerImageClusters(fname, iterations, Kclusters, patchSize, heuristicFunc);

	if (!imwrite(std::string(OUTPUT_TEST_DIR) + std::to_string(testNumber) + fname.substr(INPUT_TEST_DIR_LEN), dstImage)) {
		std::cout << "Error: Destination image not created: " << fname << "\n";
	}
	else {
		std::cout << "Destination image created: " << fname << "\n";
	}
}

void tiger_detection::threadedTestImages(int testNumber, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other), std::string heuristicFuncName) {
		
	std::pair<std::vector<std::string>, std::vector<std::string>> dirsAndFiles = file_tester::obtainFileNames();

	file_tester::makeTestDirs(testNumber, dirsAndFiles.first, iterations, Kclusters, patchSize, heuristicFuncName);

	std::cout << "Clustering images..." << "\n";

	std::vector<std::thread> threads;
	for (auto const& fileName : dirsAndFiles.second) {
		std::thread th(thFunction, testNumber, fileName, iterations, Kclusters, patchSize, heuristicFunc);
		threads.push_back(std::move(th));
	}

	for (auto& thr : threads) {
		thr.join();
	}
}

int main()
{

	// 1 - K-means clustering example
	int iterations = 0;
	int Kclusters = 0;
	int patchSize = 8;

	// 2 - Binned histogram
	int bins = 1;

	//----------------------------------------------------------------------

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - K-means clustering example\n");
		printf(" 2 - Binned histogram\n");
		printf(" 3 - Show image features\n");
		printf(" 4 - Test folder of tiger images (parallel testing)\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
			case 1:
				std::cout << "Iterations:\n";
				std::cin >> iterations;
				std::cout << "Number of clusters:\n";
				std::cin >> Kclusters;
				std::cout << "Patch size:\n";
				std::cin >> patchSize;

				tiger_detection::showClusters(iterations, Kclusters, patchSize, algorithms::cosineSimilarityHeuristic);
				break;
			case 2:
				std::cout << "Number of bins:\n";
				std::cin >> bins;

				tiger_detection::showBinnedHistogram(bins);
				break;
			case 3:
				tiger_detection::showImageFeatures();
				break;
			case 4:
				tiger_detection::threadedTestImages(1, 10, 5, 8, algorithms::cosineSimilarityHeuristic, std::string("cosine similarity"));
				break;
			default:
				break;
		}
	}
	while (op!=0);
	return 0;
}
