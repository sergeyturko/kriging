#ifndef KRIGING_H
#define KRIGING_H

#include "opencv2\opencv.hpp"

class kriging
{
public:
	kriging(size_t radiusMF = 1, float threshMF = 0.6);
	bool read(char* fname);
	void show() const;
	void write() const;
	bool calcHist();
	bool setT(unsigned char t0, unsigned char t1);
	bool calcIndicator();
	bool majorityFilter();
	bool thresholding();

	//virtual bool calcCovariance() = 0;
	bool calcCovariance();

protected:
	unsigned char m_T0;
	unsigned char m_T1;

	cv::Mat m_inputImg;
	cv::Mat m_outputImg;
	cv::Mat m_threshold;
	cv::Mat m_initialPopulation;
	cv::Mat m_indicator0;
	cv::Mat m_indicator1;
	cv::Mat m_cumProbHist;

	int m_radiusMF;
	float m_threshMF;

	long int m_numAllPixels;
	float m_sd0;				// standart diviations of the thresholded P0 pupulaton
	float m_sd1;				// standart diviations of the thresholded P1 pupulaton
};

class fixedWindowKriging : public kriging
{

};



//class indicator_kriging : public kriging
//{
//public: //protected:
//	bool treshoold() override;
//	bool majorityFilter() override;
//	void runKernelMF(int r, int c, const cv::Mat tempPopulation) override;
//	bool calcIndicator();
//	bool calcCovariance();
//	double covariance(std::vector<float> a, float suma, std::vector<float> b, float sumb);
//	void normolize(cv::Mat& x, const cv::Mat a);
//	float cumulative(float x);
//
//	void calcProbability(cv::Mat& prob, const cv::Mat kernel, int T);
//	void calcPrRun();
//
//	void setPopulation();
//	cv::Mat Population;
//	cv::Mat StartPopulation;
//	cv::Mat x0, x1;
//	cv::Mat Probability0, Probability1;
//	void save(char *fname);
//
//
//};

#endif // KGIGING_H