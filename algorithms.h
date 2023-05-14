#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <string>
#include <vector>

#define __MAX_VALUE__ 9223372036854775807
#define INSIGNIFICANT 0.00000000001


namespace algorithms {
    struct Point {
        double x, y;     // coordinates
        long long int cluster;     // no default cluster
        double minHeuristic;  // default infinite dist to nearest cluster
        std::vector<double> features;
        cv::Rect patchRect;

        Point() :
            x(0.0),
            y(0.0),
            cluster(-1),
            minHeuristic(__MAX_VALUE__) {}

        Point(double x, double y, cv::Rect rect) :
            x(x),
            y(y),
            cluster(-1),
            patchRect(rect),
            minHeuristic(__MAX_VALUE__) {}

        double heuristic(Point other, double(*heuristicFunc)(Point p, Point other)) {
            return heuristicFunc(*this, other);
        }
    };


    double euclidianHeuristic(Point p, Point other);

    double cosineSimilarityHeuristic(Point p, Point other);

    // Utility functions
    bool compareColors(Vec3b colorA, Vec3b colorB);
    std::vector<Vec3b> getRandomColors(int size);

    // Algorithms
    std::vector<Point> kMeansClustering(std::vector<Point>* points, int iterations, int Kclusters, double(*heuristicFunc)(Point p, Point other));
    std::vector<int> binnedHistogram(Mat_<uchar> src, int numberOfBins);
};

#endif
