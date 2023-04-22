#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#define __MAX_VALUE__ 9223372036854775807
#define INSIGNIFICANT 0.00000000001


namespace Algorithms {
    struct Point {
        double x, y;     // coordinates
        cv::Vec3b color;
        long long int cluster;     // no default cluster
        double minHeuristic;  // default infinite dist to nearest cluster
        bool featuresInitialized;
        std::vector<double> features;

        Point() :
            x(0.0),
            y(0.0),
            featuresInitialized(false),
            color(cv::Vec3b::all(255)),
            cluster(-1),
            minHeuristic(__MAX_VALUE__) {}

        Point(double x, double y, cv::Vec3b col) :
            x(x),
            y(y),
            featuresInitialized(false),
            color(col),
            cluster(-1),
            minHeuristic(__MAX_VALUE__) {}

        void initFeatures() {
            double xy_magnitude = max(sqrt(x * x + y * y), INSIGNIFICANT);
            double rgb_magnitude = max(sqrt(color[2] * color[2] + color[1] * color[1] + color[0] * color[0]), INSIGNIFICANT);
            //// x
            //features.push_back((double)x / xy_magnitude);

            //// y
            //features.push_back((double)y / xy_magnitude);

            // Red
            features.push_back((double)color[2] / rgb_magnitude);

            // Green
            features.push_back((double)color[1] / rgb_magnitude);

            // Blue
            features.push_back((double)color[0] / rgb_magnitude);
        }

        double heuristic(Point p) {
            double featuresCoefficient = 0.0;

            if (!featuresInitialized) {
                initFeatures();
                featuresInitialized = true;
            }

            if (!p.featuresInitialized) {
                p.initFeatures();
                p.featuresInitialized = true;
            }

            for (int i = 0; i < features.size(); i++) {
                double feature = features[i];
                double featureOther = p.features[i];

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
    };


    // Utility functions
    bool compareColors(Vec3b colorA, Vec3b colorB);
    std::vector<Vec3b> getRandomColors(int size);

    // Algorithms
    std::vector<Algorithms::Point> kMeansClustering(std::vector<Algorithms::Point>* points, int iterations, int Kclusters);
    std::vector<int> binnedHistogram(Mat_<uchar> src, int numberOfBins);
};

#endif
