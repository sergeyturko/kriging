#ifndef KRIGING_H
#define KRIGING_H

#include "opencv2\opencv.hpp"

class kriging
{
public:
	kriging(float mT = 0.6, int rmf = 1, int rf = 2); // TODO radiuses
	bool read(char* fname);
	bool calcHist();
	bool setT(uchar t0, uchar t1);
	virtual bool majorityFilter() = 0;
	virtual bool treshoold() = 0;
	virtual bool calcIndicator() = 0;
protected:
	virtual void runKernelMF(int r, int c, const cv::Mat tempPopulation) = 0;
	int Hist[256];
	float probHist[256];
	cv::Mat CurrentImages;
	cv::Mat indicator0, indicator1;
	int T1, T0;
	float sd0, sd1;
	int radiusKernelMF;
	float majorityTresh;
	int radiusF;
	float Covariance[28];

};

class indicator_kriging : public kriging
{
public: //protected:
	bool treshoold() override;
	bool majorityFilter() override;
	void runKernelMF(int r, int c, const cv::Mat tempPopulation) override;
	bool calcIndicator();
	bool calcCovariance();
	double covariance(std::vector<float> a, float suma, std::vector<float> b, float sumb);
	void normolize(cv::Mat& x, const cv::Mat a);
	float cumulative(float x);

	void calcProbability(cv::Mat& prob, const cv::Mat kernel, int T);
	void calcPrRun();

	void setPopulation();
	cv::Mat Population;
	cv::Mat StartPopulation;
	cv::Mat x0, x1;
	cv::Mat Probability0, Probability1;
	void save(char *fname);


};

#endif // KGIGING_H