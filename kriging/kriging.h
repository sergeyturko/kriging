#ifndef KRIGING_H
#define KRIGING_H

#include <fstream>

#include "opencv2\opencv.hpp"


class kriging
{
public:
	kriging(int radiusMF = 1, float threshMF = 0.6);
	bool read(const cv::String& fname);
	void show() const;
	bool calcHist();
	bool setT(unsigned char t0, unsigned char t1);
	bool calcIndicator();
	bool majorityFilter();
	bool thresholding();

	virtual void write(const cv::String& imgName) = 0;
	virtual bool calcCovarianceMatrix() = 0;
	virtual bool calcProbability() = 0;

protected:
	unsigned char m_T0;
	unsigned char m_T1;

	cv::Mat m_inputImg;
	cv::Mat m_threshold; // output
	cv::Mat m_initialPopulation;
	cv::Mat m_indicator0;
	cv::Mat m_indicator1;
	cv::Mat m_cumProbHist;
	cv::Mat m_probabilityPopulation0;
	cv::Mat m_probabilityPopulation1;

	int m_radiusMF;
	float m_threshMF;

	long int m_numAllPixels;
	float m_sd0;				// standart diviations of the thresholded P0 pupulaton
	float m_sd1;				// standart diviations of the thresholded P1 pupulaton

	void escapeNegativeWeights(cv::Mat& weightMatrix, const cv::Mat& krigingSystemRight) const;
};


class fixedWindowKriging : public kriging
{
public:
	cv::Mat m_krigingKernel0;
	cv::Mat m_krigingKernel1;

	std::vector<std::pair<int, int>> m_krigingKernelIndex; // pair(row, col)
	int m_numElemUnderWindow;
	int m_radiusKrigng;

	fixedWindowKriging(int radiusKriging = 3, int radiusMF = 1, float threshMF = 0.6);
	void write(const cv::String& imgName) override;
	bool calcCovarianceMatrix() override;
	bool calcProbability() override; // TODO optimize

private: 
	void setKernelIndexArray();
	cv::Mat getKrigingSystem(const cv::Mat& sequencesMatrix, bool left_right) const; // false - return left matrix; true - return right matrix
	float covariance(const cv::Mat seq0, const cv::Mat seq1) const;
	cv::Mat getKrigignKernel(const cv::Mat& weightsMatrix);
};


#endif // KGIGING_H