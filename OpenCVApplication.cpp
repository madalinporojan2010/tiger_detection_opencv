// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "algorithms.h"


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("opened image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

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


void showClusters(int iterations, int Kclusters) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat_<Vec3b> dst(height, width);
		dst.setTo(cv::Scalar(255, 255, 255));


		std::vector<Algorithms::Point> points;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (!Algorithms::compareColors(Vec3b(src(i, j), 0, 0), Vec3b(255, 0, 0))) {
					points.push_back(Algorithms::Point((double)j, (double)i));
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

		imshow("original image (grayscale)", src);
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

int main()
{

	// 4 - K-means clustering example
	int iterations = 0;
	int Kclusters = 0;

	// 5- Binned histogram
	int bins = 1;

	//----------------------------------------------------------------------

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Basic image opening...\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Color to Gray\n");
		printf(" 4 - K-means clustering example\n");
		printf(" 5 - Binned histogram\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testColor2Gray();
				break;
			case 4:
				std::cout << "Iterations:\n";
				std::cin >> iterations;
				std::cout << "Number of clusters:\n";
				std::cin >> Kclusters;

				showClusters(iterations, Kclusters);
				break;
			case 5:
				std::cout << "Number of bins:\n";
				std::cin >> bins;

				showBinnedHistogram(bins);
				break;
		}
	}
	while (op!=0);
	return 0;
}
