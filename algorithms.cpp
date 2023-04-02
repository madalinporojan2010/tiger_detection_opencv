#include "stdafx.h"
#include "common.h"
#include "algorithms.h"
#include <random>

std::default_random_engine gen;
std::uniform_int_distribution<int> d(0, 255);

std::vector<Algorithms::Point> Algorithms::kMeansClustering(std::vector<Algorithms::Point> *points, int iterations, int Kclusters) {
    // initializing the clusters
    std::vector<Algorithms::Point> centroids;
    srand(time(0));
    for (int i = 0; i < Kclusters; i++) {
        centroids.push_back(points->at(rand() % points->size()));
    }

    for (int i = 0; i < iterations; i++) {
        // assigning points to a cluster
        for (std::vector<Algorithms::Point>::iterator centroidIt = std::begin(centroids); centroidIt != std::end(centroids); centroidIt++) {
            int clusterId = centroidIt - begin(centroids);

            for (std::vector<Algorithms::Point>::iterator pointIt = points->begin(); pointIt != points->end(); pointIt++) {
                Algorithms::Point point = *pointIt;
                double distance = centroidIt->distance(point);
                if (distance < point.minDistance) {
                    point.minDistance = distance;
                    point.cluster = clusterId;
                }
                *pointIt = point;
            }
        }

        // computing new centroids
        std::vector<int> nPoints;
        std::vector<double> sumX, sumY; // is fine?

        for (int j = 0; j < Kclusters; ++j) {
            nPoints.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
        }

        for (std::vector<Algorithms::Point>::iterator pointIt = points->begin(); pointIt != points->end(); pointIt++) {
            int clusterId = pointIt->cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += pointIt->x;
            sumY[clusterId] += pointIt->y;

            pointIt->minDistance = __MAX_VALUE__;
        }

        for (std::vector<Algorithms::Point>::iterator centroidIt = std::begin(centroids); centroidIt != std::end(centroids); centroidIt++) {
            int clusterId = centroidIt - std::begin(centroids);
            centroidIt->x = sumX[clusterId] / nPoints[clusterId];
            centroidIt->y = sumY[clusterId] / nPoints[clusterId];
        }
    }

    return centroids;
}

bool Algorithms::compareColors(Vec3b colorA, Vec3b colorB) {
    return colorA[0] == colorB[0] && colorA[1] == colorB[1] && colorA[2] == colorB[2];
}

std::vector<Vec3b> Algorithms::getRandomColors(int size) {
    std::vector<Vec3b> colors(size, Vec3b(255, 255, 255));

    for (int i = 0; i < size; i++) {
        uchar b = d(gen);
        uchar g = d(gen);
        uchar r = d(gen);
        colors[i] = Vec3b(b, g, r);
    }

    return colors;
}