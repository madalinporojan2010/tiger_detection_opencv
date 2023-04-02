#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#define __MAX_VALUE__ 9999999999.9


namespace Algorithms {
    struct Point {
        double x, y;     // coordinates
        int cluster;     // no default cluster
        double minDistance;  // default infinite dist to nearest cluster

        Point() :
            x(0.0),
            y(0.0),
            cluster(-1),
            minDistance(__MAX_VALUE__) {}

        Point(double x, double y) :
            x(x),
            y(y),
            cluster(-1),
            minDistance(__MAX_VALUE__) {}

        double distance(Point p) {
            return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
        }
    };


    // Utility functions
    bool compareColors(Vec3b colorA, Vec3b colorB);
    std::vector<Vec3b> getRandomColors(int size);

    // Algorithms
    std::vector<Algorithms::Point> kMeansClustering(std::vector<Algorithms::Point>* points, int iterations, int Kclusters);
};

#endif
