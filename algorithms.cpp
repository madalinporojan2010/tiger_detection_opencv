#include "stdafx.h"
#include "common.h"
#include "algorithms.h"
#include <random>

std::default_random_engine gen;
std::uniform_int_distribution<int> d(0, 255);

std::vector<algorithms::Point> algorithms::kMeansClustering(std::vector<algorithms::Point> *points, int iterations, int Kclusters, double(*heuristicFunc)(algorithms::Point p, algorithms::Point other)) {
    // initializing the clusters
    std::vector<algorithms::Point> centroids;
    srand(time(0));
    for (int i = 0; i < Kclusters; i++) {
        centroids.push_back(points->at(rand() % points->size()));
    }

    for (int i = 0; i < iterations; i++) {
        // assigning points to a cluster
        for (std::vector<algorithms::Point>::iterator centroidIt = std::begin(centroids); centroidIt != std::end(centroids); centroidIt++) {
            long long int clusterId = centroidIt - begin(centroids);

            for (std::vector<algorithms::Point>::iterator pointIt = points->begin(); pointIt != points->end(); pointIt++) {
                algorithms::Point point = *pointIt;
                double heuristic = centroidIt->heuristic(point, heuristicFunc);
                if (heuristic < point.minHeuristic) {
                    point.minHeuristic = heuristic;
                    point.cluster = clusterId;
                }
                *pointIt = point;
            }
        }

        // computing new centroids
        std::vector<int> nPoints;
        std::vector<double> sumX, sumY;

        for (int j = 0; j < Kclusters; ++j) {
            nPoints.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
        }

        for (std::vector<algorithms::Point>::iterator pointIt = points->begin(); pointIt != points->end(); pointIt++) {
            long long int clusterId = pointIt->cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += pointIt->x;
            sumY[clusterId] += pointIt->y;

            pointIt->minHeuristic = __MAX_VALUE__;
        }

        for (std::vector<algorithms::Point>::iterator centroidIt = std::begin(centroids); centroidIt != std::end(centroids); centroidIt++) {
            int clusterId = centroidIt - std::begin(centroids);
            centroidIt->x = sumX[clusterId] / nPoints[clusterId];
            centroidIt->y = sumY[clusterId] / nPoints[clusterId];
        }

    }

    return centroids;
}

std::vector<int> algorithms::binnedHistogram(Mat_<uchar> src, int numberOfBins) {
    int height = src.rows;
    int width = src.cols;

    std::vector<int> hist(numberOfBins);
    std::fill(hist.begin(), hist.end(), 0);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (src(i, j) / (256 / numberOfBins) > numberOfBins - 1) {
                hist[numberOfBins - 1]++;
            }
            else {
                hist[src(i, j) / (256 / numberOfBins)]++;
            }
        }
    }
    return hist;
}

bool algorithms::compareColors(Vec3b colorA, Vec3b colorB) {
    return colorA[0] == colorB[0] && colorA[1] == colorB[1] && colorA[2] == colorB[2];
}

std::vector<Vec3b> algorithms::getRandomColors(int size) {
    std::vector<Vec3b> colors(size, Vec3b(255, 255, 255));

    for (int i = 0; i < size; i++) {
        uchar b = d(gen);
        uchar g = d(gen);
        uchar r = d(gen);
        colors[i] = Vec3b(b, g, r);
    }

    return colors;
}

double algorithms::euclidianHeuristic(Point p, Point other) {
    double featuresCoefficient = 0.0;
    for (int i = 0; i < p.features.size(); i++) {
        double feature = p.features[i];
        double featureOther = other.features[i];

        if (isnan(feature)) {
            feature = INSIGNIFICANT;
        }

        if (isnan(featureOther)) {
            featureOther = INSIGNIFICANT;
        }

        featuresCoefficient += (feature - featureOther) * (feature - featureOther);
    }
    // min or max
    return featuresCoefficient;
}

double algorithms::cosineSimilarityHeuristic(Point p, Point other) {
    double s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int i = 0; i < p.features.size(); i++) {
        s1 += p.features.at(i) * other.features.at(i);
        s2 += p.features.at(i) * p.features.at(i);
        s3 += other.features.at(i) * other.features.at(i);
    }
    return 1-(s1 / (sqrt(s2) * sqrt(s3)));
}