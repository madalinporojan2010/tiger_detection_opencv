// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "algorithms.h"
#include "haralick_feat.h"

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> dst(height, width);

		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("original image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void showHistogram(const std::string& name, std::vector<int> hist, const int  hist_cols, const int hist_height) {
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


std::vector<double> getPatchFeatures(cv::Mat pointPatch) {
	std::vector<int> deltax({ 1 });
	std::vector<int> deltay({ 0 });

	HaralickExtractor extract;
	std::vector<double> allFeatures = extract.getFeaturesFromImage(pointPatch, deltax, deltay, false);

	std::vector<double> selectedFeatures;


	// Energy:
	selectedFeatures.push_back(allFeatures[0]);

	// Entropy:
	selectedFeatures.push_back(allFeatures[1]);

	// Inverse Difference Moment:
	selectedFeatures.push_back(allFeatures[2]);

	// Info Measure of Correlation 2:
	selectedFeatures.push_back(allFeatures[5]);

	// Contrast:
	selectedFeatures.push_back((double)allFeatures[6] / 255);

	return selectedFeatures;
}


std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> createPatches(Mat src, int patchSize) {
	int height = src.rows;
	int width = src.cols;

	std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> patches;

	cv::Size patch_size(patchSize, patchSize); // Specify the patch size

	for (int i = 0; i < src.rows; i += patch_size.height) {
		for (int j = 0; j < src.cols; j += patch_size.width) {
			cv::Rect patch_rect(j, i, min(patch_size.width, width - j - 1), min(patch_size.height, height - i - 1));
			cv::Mat patch = src(patch_rect);
			patches.push_back(std::make_tuple(patch_rect, patch, getPatchFeatures(patch)));
		}
	}
	return patches;
}

void showClusters(int iterations, int Kclusters, int patchSize) {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<Vec3b> src_color = imread(fname);

		int height = src.rows;
		int width = src.cols;
		if (patchSize > max(height, width) || patchSize <= 0) {
			patchSize = max(height, width);
		}

		Mat_<Vec3b> dst(height, width);
		dst.setTo(cv::Scalar(255, 255, 255));

		std::vector<Algorithms::Point> points;


		std::vector<std::tuple<cv::Rect, cv::Mat, std::vector<double>>> patches = createPatches(src, patchSize);

		for (const auto patch : patches) {
			for (int i = std::get<0>(patch).y; i < std::get<0>(patch).y + std::get<0>(patch).height; i++) {
				for (int j = std::get<0>(patch).x; j < std::get<0>(patch).x + std::get<0>(patch).width; j++) {
					Algorithms::Point point = Algorithms::Point((double)j, (double)i, src_color(i, j));
					point.features = std::get<2>(patch);

					points.push_back(point);
					
				}
			}
		}

		std::vector<Algorithms::Point> centroids;
		centroids = Algorithms::kMeansClustering(&points, iterations, Kclusters);

		int markSize = 20;
		int markThickness = 2;
		cv::Scalar markColor = cv::Scalar(0, 0, 0);

		std::vector<int> clusterIds;
		std::transform(points.begin(), points.end(), std::back_inserter(clusterIds), [](Algorithms::Point x) {
			return x.cluster;
		});

		int maxClusterId = *std::max_element(clusterIds.begin(), clusterIds.end());

		std::vector<Vec3b> randomColors = Algorithms::getRandomColors(maxClusterId + 1);

		for (auto const &point : points) {
			dst((int)point.y, (int)point.x) = randomColors[point.cluster];
		}

		for (auto const& centroid : centroids) {
			// mark the centroids with a plus symbol
			cv::line(dst, cv::Point(centroid.x - markSize, centroid.y), cv::Point(centroid.x + markSize, centroid.y), markColor, markThickness);
			cv::line(dst, cv::Point(centroid.x, centroid.y - markSize), cv::Point(centroid.x, centroid.y + markSize), markColor, markThickness);
		}

		//patches outlines

		cv::Scalar rectanglesColor(250, 250, 250);


		imshow("original image", src_color);
		imshow("clustered image", dst);
		waitKey();
	}
}

void showBinnedHistogram(int numberOfBins) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		double t = (double)getTickCount(); // Get the current time [s]

		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		std::vector<int> hist = Algorithms::binnedHistogram(src, numberOfBins);

		std::vector<int> showedHist;

		for (int h : hist) {
			for (int i = 0; i < 256 / hist.size(); i++) {
				showedHist.push_back(h);
			}
		}

		showHistogram("Binned histogram", showedHist, showedHist.size(), 100);
		waitKey();
	}
}

void showImageFeatures() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		double t = (double)getTickCount(); // Get the current time [s]

		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);


		std::vector<int> deltax({ 1 });
		std::vector<int> deltay({ 0 });

		HaralickExtractor extract; 
		extract.getFeaturesFromImage(src, deltax, deltay, true);

		waitKey();
	}
}


int main()
{

	// 2 - K-means clustering example
	int iterations = 0;
	int Kclusters = 0;
	int patchSize = 8;

	// 3 - Binned histogram
	int bins = 1;

	//----------------------------------------------------------------------

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Color to Gray\n");
		printf(" 2 - K-means clustering example\n");
		printf(" 3 - Binned histogram\n");
		printf(" 4 - Show image features\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
			case 1:
				testColor2Gray();
				break;
			case 2:
				std::cout << "Iterations:\n";
				std::cin >> iterations;
				std::cout << "Number of clusters:\n";
				std::cin >> Kclusters;
				std::cout << "Patch size:\n";
				std::cin >> patchSize;

				showClusters(iterations, Kclusters, patchSize);
				break;
			case 3:
				std::cout << "Number of bins:\n";
				std::cin >> bins;

				showBinnedHistogram(bins);
				break;
			case 4:
				showImageFeatures();
				break;
			default:
				break;
		}
	}
	while (op!=0);
	return 0;
}
