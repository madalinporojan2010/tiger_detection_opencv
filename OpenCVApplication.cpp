#include "stdafx.h"
#include "common.h"
#include "algorithms.h"
#include "OpenCVApplication.h"
#include "file_tester.h"
#include <random>
#include <functional>

std::random_device dev;
std::mt19937 rng(dev());

// Function for showing the histogram in a visual form
void Segmentation::showHistogram(const std::string& name, std::vector<int> hist, const int  hist_cols, const int hist_height) {
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

// Function for creating patches of a given size, from a given Mat src
std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> Segmentation::createPatches(Mat src_color, Mat src, Mat src_hue, int patchSize) {
	int height = src.rows;
	int width = src.cols;

	std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> patches;

	cv::Size patch_size(patchSize, patchSize); // Specify the patch size

	for (int i = 0; i < src.rows; i += patch_size.height) {
		for (int j = 0; j < src.cols; j += patch_size.width) {
			cv::Rect patch_rect(j, i, min(patch_size.width, width - j - 1), min(patch_size.height, height - i - 1));
			cv::Mat patch = src(patch_rect);

			std::vector<double> features = std::vector({ src_color.at<Vec3b>(i,j)[0] * 1.0, src_color.at<Vec3b>(i,j)[1] * 1.0, src_color.at<Vec3b>(i,j)[2] * 1.0 });
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

// Function for showing the clusters of an image in a visual form 
void Segmentation::showClusters(int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other)) {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src_color = imread(fname);
		Mat_<Vec3b> dstHSV = Segmentation::computeImageClusters(fname, "HSV", iterations, Kclusters, patchSize, heuristicFunc);
		Mat_<Vec3b> dstLab = Segmentation::computeImageClusters(fname, "Lab", iterations, Kclusters, patchSize, heuristicFunc);
		Mat_<Vec3b> dstYUV = Segmentation::computeImageClusters(fname, "YUV", iterations, Kclusters, patchSize, heuristicFunc);
		Mat_<Vec3b> dstYCrCb = Segmentation::computeImageClusters(fname, "YCrCb", iterations, Kclusters, patchSize, heuristicFunc);

		imshow("original image", src_color);
		imshow("HSV", dstHSV);
		imshow("Lab", dstLab);
		imshow("YUV", dstYUV);
		imshow("YCrCb", dstYCrCb);

		waitKey();
	}
}

// Function for showing the binned histogram in a visual form
void Segmentation::showBinnedHistogram(int numberOfBins) {
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
		Segmentation::showHistogram("Binned Histogram for given patch", showedHist, showedHist.size(), 100);
		waitKey();
	}
}

Mat cvtToReadType(Mat src, const std::string readType) {
	Mat dst;
	if (readType == "HSV") {
		cv::cvtColor(src, dst, COLOR_BGR2HSV);
		return dst;
	}
	else if (readType == "Lab") {
		cv::cvtColor(src, dst, COLOR_BGR2Lab);
		return dst;
	}
	else if (readType == "YUV") {
		cv::cvtColor(src, dst, COLOR_BGR2YUV);
		return dst;
	}
	else if (readType == "YCrCb") {
		cv::cvtColor(src, dst, COLOR_BGR2YCrCb);
		return dst;
	}
	return src;
}

// Image clustering funcion
cv::Mat_<Vec3b> Segmentation::computeImageClusters(const cv::String fname, const std::string readType, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other)) {

	Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
	Mat_<Vec3b> src_color = imread(fname);
	Mat_<Vec3b> src_read_color = cvtToReadType(src_color, readType);
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


	std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> patches = Segmentation::createPatches(src_read_color, src, src_hue, patchSize);

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

// Function used for the testing threads
void thThreadedTestImages(int testNumber, std::string fname, const std::string readType, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other)) {
	Mat_<Vec3b> dstImage = Segmentation::computeImageClusters(fname, readType, iterations, Kclusters, patchSize, heuristicFunc);

	if (!imwrite(std::string(OUTPUT_TEST_DIR) + std::to_string(testNumber) + fname.substr(INPUT_TEST_DIR_LEN), dstImage)) {
		std::cout << "Error: " << "Test #" << std::to_string(testNumber) << ": Destination image not created: " << fname << "\n";
	}
	else {
		std::cout << "Test #" << std::to_string(testNumber) << ": Destination image created: " << fname << "\n";
	}
}

// Function used for testing a single file (generating the image clusters)
std::vector<std::thread> Segmentation::threadedTestImages(int testNumber, const std::string readType, int iterations, int Kclusters, int patchSize, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other), std::string heuristicFuncName) {

	std::pair<std::vector<std::string>, std::vector<std::string>> dirsAndFiles = file_tester::obtainFileNames();

	file_tester::makeTestDirs(testNumber, dirsAndFiles.first, iterations, Kclusters, patchSize, heuristicFuncName);

	std::cout << "Test #" << std::to_string(testNumber) << ": Clustering images..." << "\n";

	std::vector<std::thread> threads;
	for (auto const& fileName : dirsAndFiles.second) {
		std::thread th(thThreadedTestImages, testNumber, fileName, readType, iterations, Kclusters, patchSize, heuristicFunc);
		threads.push_back(std::move(th));
	}

	return threads;
}

// Function used for testing multiple files with randomized arguments values
void Segmentation::randomizedTesting(int numberOfTests, const std::string readType) {
	std::uniform_int_distribution<std::mt19937::result_type> distIterations(15, 60);
	std::uniform_int_distribution<std::mt19937::result_type> distKclusters(2, 20);

	std::vector<int> defaultPatchSizes{ 8, 16 };
	std::uniform_int_distribution<std::mt19937::result_type> distPatchSize(0, 1);
	std::uniform_int_distribution<std::mt19937::result_type> distFunctionNumber(0, 1);

	std::vector<std::thread> threads;

	for (int i = 1; i <= numberOfTests; i++) {
		std::vector<std::thread> localThreads;
		std::cout << "Starting test #" << i << "\n";
		int funcNum = distFunctionNumber(rng);
		if (funcNum == 0) {
			localThreads = Segmentation::threadedTestImages(i, readType, distIterations(rng), distKclusters(rng), defaultPatchSizes[distPatchSize(rng)], algorithms::euclidianHeuristic, "Euclidian Distance");
		}
		else {
			localThreads = Segmentation::threadedTestImages(i, readType, distIterations(rng), distKclusters(rng), defaultPatchSizes[distPatchSize(rng)], algorithms::cosineSimilarityHeuristic, "Cosine Similarity Distance");
		}

		for (auto& th : localThreads) {
			threads.push_back(std::move(th));
		}
	}

	for (auto& th : threads) {
		th.join();
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

	// 5 - Number of tests
	int numberOfTests = 1;

	//----------------------------------------------------------------------

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - K-means clustering example\n");
		printf(" 2 - Binned histogram\n");
		printf(" 4 - EXAMPLE: Test folder of tiger images (parallel testing)\n");
		printf(" 5 - Test folder of tiger images (parallel testing)\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		if (scanf("%d", &op) == 1) {
			switch (op)
			{
			case 1:
				std::cout << "Iterations:\n";
				std::cin >> iterations;
				std::cout << "Number of clusters:\n";
				std::cin >> Kclusters;
				std::cout << "Patch size:\n";
				std::cin >> patchSize;

				Segmentation::showClusters(iterations, Kclusters, patchSize, algorithms::euclidianHeuristic);
				break;
			case 2:
				std::cout << "Number of bins:\n";
				std::cin >> bins;

				Segmentation::showBinnedHistogram(bins);
				break;
			case 4:
				Segmentation::threadedTestImages(1, "HSV", 10, 5, 8, algorithms::euclidianHeuristic, std::string("euclidiane heuristic"));
				break;
			case 5:
				std::cout << "Number of randomized tests: \n";
				std::cin >> numberOfTests;

				Segmentation::randomizedTesting(numberOfTests, "HSV");

				break;
			default:
				break;
			}
		}
	} while (op != 0);
	return 0;
}
