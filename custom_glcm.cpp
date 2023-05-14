#include "stdafx.h"
#include "common.h"
#include "custom_glcm.h"
#include <vector>

#define EPS 0.00000001

double custom_glcm::Entropy(std::vector<double> vec) {
    double result = 0.0;
    for (int i = 0; i < vec.size(); i++)
        result += vec[i] * log(vec[i] + EPS);
    return -1 * result;
}

void custom_glcm::meanStd(std::vector<double> v, double& m, double& stdev) {
    double sum = 0.0;
    std::for_each(std::begin(v), std::end(v), [&](const double d) {
        sum += d;
        });
    m = sum / v.size();

    double accum = 0.0;
    std::for_each(std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
        });

    stdev = sqrt(accum / (v.size() - 1));
}

//Marginal probabilities as in px = sum on j(p(i, j))
//                             py = sum on i(p(i, j))
std::vector<double> custom_glcm::MargProbx(cv::Mat cooc) {
    std::vector<double> result(cooc.rows, 0.0);
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            result[i] += cooc.at<double>(i, j);
    return result;
}

std::vector<double> custom_glcm::MargProby(cv::Mat cooc) {
    std::vector<double> result(cooc.cols, 0.0);
    for (int j = 0; j < cooc.cols; j++)
        for (int i = 0; i < cooc.rows; i++)
            result[j] += cooc.at<double>(i, j);
    return result;
}

//probsum  := Px+y(k) = sum(p(i,j)) given that i + j = k
std::vector<double> custom_glcm::ProbSum(cv::Mat cooc) {
    std::vector<double> result(cooc.rows * 2, 0.0);
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            result[i + j] += cooc.at<double>(i, j);
    return result;
}

//probdiff := Px-y(k) = sum(p(i,j)) given that |i - j| = k
std::vector<double> custom_glcm::ProbDiff(cv::Mat cooc) {
    std::vector<double> result(cooc.rows, 0.0);
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            result[abs(i - j)] += cooc.at<double>(i, j);
    return result;
}
/*Features from coocurrence cv::Matrix*/
double custom_glcm::HaralickEnergy(cv::Mat cooc) {
    double energy = 0;
    for (int i = 0; i < cooc.rows; i++) {
        for (int j = 0; j < cooc.cols; j++) {
            energy += cooc.at<double>(i, j) * cooc.at<double>(i, j);
        }
    }
    return energy;
}

double custom_glcm::HaralickEntropy(cv::Mat cooc) {
    double entrop = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            entrop += cooc.at<double>(i, j) * log(cooc.at<double>(i, j) + EPS);
    return -1 * entrop;
}

double custom_glcm::HaralickInverseDifference(cv::Mat cooc) {
    double res = 0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            res += cooc.at<double>(i, j) * (1 / (1 + (i - j) * (i - j)));
    return res;
}

/*Features from MargProbs */
double custom_glcm::HaralickCorrelation(cv::Mat cooc, std::vector<double> probx, std::vector<double> proby) {
    double corr = 0.0;
    double meanx = 0.0, meany = 0.0, stddevx = 0.0, stddevy = 0.0;
    meanStd(probx, meanx, stddevx);
    meanStd(proby, meany, stddevy);
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            corr += (i * j * cooc.at<double>(i, j)) - meanx * meany;
    return corr / (stddevx * stddevy);
}

//InfoMeasure1 = HaralickEntropy - HXY1 / max(HX, HY)
//HXY1 = sum(sum(p(i, j) * log(px(i) * py(j))
double custom_glcm::HaralickInfoMeasure1(cv::Mat cooc, double ent, std::vector<double> probx, std::vector<double> proby) {
    double hx = Entropy(probx);
    double hy = Entropy(proby);
    double hxy1 = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            hxy1 += cooc.at<double>(i, j) * log(probx[i] * proby[j] + EPS);
    hxy1 = -1 * hxy1;

    return (ent - hxy1) / max(hx, hy);

}

//InfoMeasure2 = sqrt(1 - exp(-2(HXY2 - HaralickEntropy)))
//HX2 = sum(sum(px(i) * py(j) * log(px(i) * py(j))
double custom_glcm::HaralickInfoMeasure2(cv::Mat cooc, double ent, std::vector<double> probx, std::vector<double> proby) {
    double hxy2 = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            hxy2 += probx[i] * proby[j] * log(probx[i] * proby[j] + EPS);
    hxy2 = -1 * hxy2;

    return sqrt(1 - exp(-2 * (hxy2 - ent)));
}

/*Features from ProbDiff*/
double custom_glcm::HaralickContrast(cv::Mat cooc, std::vector<double> diff) {
    double contrast = 0.0;
    for (int i = 0; i < diff.size(); i++)
        contrast += i * i * diff[i];
    return contrast/ diff.size()/5.0;
}

double custom_glcm::HaralickDiffEntropy(cv::Mat cooc, std::vector<double> diff) {
    double diffent = 0.0;
    for (int i = 0; i < diff.size(); i++)
        diffent += diff[i] * log(diff[i] + EPS);
    return -1 * diffent;
}

double custom_glcm::HaralickDiffVariance(cv::Mat cooc, std::vector<double> diff) {
    double diffvar = 0.0;
    double diffent = HaralickDiffEntropy(cooc, diff);
    for (int i = 0; i < diff.size(); i++)
        diffvar += (i - diffent) * (i - diffent) * diff[i];
    return diffvar;
}

/*Features from Probsum*/
double custom_glcm::HaralickSumAverage(cv::Mat cooc, std::vector<double> sumprob) {
    double sumav = 0.0;
    for (int i = 0; i < sumprob.size(); i++)
        sumav += i * sumprob[i];
    return sumav;
}

double custom_glcm::HaralickSumEntropy(cv::Mat cooc, std::vector<double> sumprob) {
    double sument = 0.0;
    for (int i = 0; i < sumprob.size(); i++)
        sument += sumprob[i] * log(sumprob[i] + EPS);
    return -1 * sument;
}

double custom_glcm::HaralickSumVariance(cv::Mat cooc, std::vector<double> sumprob) {
    double sumvar = 0.0;
    double sument = HaralickSumEntropy(cooc, sumprob);
    for (int i = 0; i < sumprob.size(); i++)
        sumvar += (i - sument) * (i - sument) * sumprob[i];
    return sumvar;
}


cv::Mat custom_glcm::MatCooc(cv::Mat img, int N, int deltax, int deltay)
{
    int atual, vizinho;
    int newi, newj;
    cv::Mat ans = cv::Mat::zeros(N + 1, N + 1, CV_64F);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            newi = i + deltay;
            newj = j + deltax;
            if (newi < img.rows && newj < img.cols && newj >= 0 && newi >= 0) {
                atual = (int)img.at<uchar>(i, j);
                vizinho = (int)img.at<uchar>(newi, newj);
                ans.at<double>(atual, vizinho) += 1.0;
            }
        }
    }
    return ans / (img.rows * img.cols);
}

//Assume tamanho deltax == tamanho deltay 
cv::Mat custom_glcm::MatCoocAdd(cv::Mat img, int N, std::vector<int> deltax, std::vector<int> deltay)
{
    cv::Mat ans, nextans;
    ans = MatCooc(img, N, deltax[0], deltay[0]);
    for (int i = 1; i < deltax.size(); i++) {
        nextans = MatCooc(img, N, deltax[i], deltay[i]);
        add(ans, nextans, ans);
    }
    return ans;
}

std::vector<double> custom_glcm::getFeatures(cv::Mat_<uchar> img) {
    std::vector<double> features;
    std::vector<int> deltax{ 1 };
    std::vector<int> deltay{ 0 };
    cv::Mat ans = MatCoocAdd(img, 255, deltax, deltay);

    std::vector<double> probx = MargProbx(ans);
    std::vector<double> proby = MargProby(ans);
    std::vector<double> diff = ProbDiff(ans);

    // Entropy
    features.push_back(HaralickEntropy(ans));
    // Energy
    features.push_back(HaralickEnergy(ans));
    // Correlation
    features.push_back(HaralickCorrelation(ans, probx, proby));
    // Contrast
    features.push_back(HaralickContrast(ans, diff));
    // Inverse difference
    features.push_back(HaralickInverseDifference(ans));
    return features;
}
