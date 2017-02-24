#include "kriging.h"
#define Population0 0
#define Population1 150
#define UnknowPopulation 95

kriging::kriging(size_t radiusMF, float threshMF) : m_radiusMF(radiusMF), m_threshMF(threshMF)
{}

bool kriging::read(char* fname)
{
	m_inputImg = cv::imread(fname);
	if (!m_inputImg.data)
		return false;
	if (m_inputImg.channels() == 3)
		cv::cvtColor(m_inputImg, m_inputImg, CV_RGB2GRAY);
	else
		return false;
	m_numAllPixels = m_inputImg.rows * m_inputImg.cols;
	return true;
}

bool kriging::calcHist()
{
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::Mat hist;

	cv::calcHist(&m_inputImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
	m_cumProbHist = cv::Mat(hist.rows, hist.cols, CV_32FC1);
	float sum = 0.0;
	for (int i = 0; i < 256; ++i)
	{
		float prob = (hist.at<float>(i, 0) / m_numAllPixels);
		m_cumProbHist.at<float>(i, 0) = prob + sum;
		sum += prob;
	}
	
	return true;
}

//bool kriging::calcHist()
//{
//	if (CurrentImages.type() != CV_8UC1)
//		return false;
//	memset(Hist, 0, sizeof(int)* 256);
//	for (int r = 0; r < CurrentImages.rows; ++r)
//	{
//		for (int c = 0; c < CurrentImages.cols; ++c)
//			++Hist[CurrentImages.at<uchar>(r, c)];
//	}
//	float N = CurrentImages.cols * CurrentImages.rows * 1.0;
//	for (int i = 0; i < 256; ++i)
//		probHist[i] = Hist[i] / N;
//	return true;
//}

bool kriging::setT(uchar t0, uchar t1)
{
	m_T1 = std::max(t0, t1);
	m_T0 = std::min(t0, t1);
	return true;
}

//bool indicator_kriging::treshoold()
//{
//	if (CurrentImages.rows == 0)
//		return false;
//
//	float mean0 = 0;
//	float mean1 = 0;
//
//	Population = cv::Mat(CurrentImages.rows, CurrentImages.cols, CurrentImages.type());
//
//	for (int i = 0; i < CurrentImages.rows; ++i)
//	{
//		for (int j = 0; j < CurrentImages.cols; ++j)
//		{
//			if (CurrentImages.at<uchar>(i, j) < T0 && CurrentImages.at<uchar>(i, j) < T1)
//			{
//				Population.at<uchar>(i, j) = Population0;
//				++mean0;
//			}
//			else if (CurrentImages.at<uchar>(i, j) > T1)
//			{
//				Population.at<uchar>(i, j) = Population1;
//				++mean1;
//			}
//			else
//				Population.at<uchar>(i, j) = UnknowPopulation;
//		}
//	}
//
//	float pix = CurrentImages.rows * CurrentImages.cols * 1.0;
//	int count0 = mean0;
//	int count1 = mean1;
//	mean0 = mean0 / pix;
//	mean1 = mean1 / pix;
//	sd0 = std::sqrtf(((((1.0 - mean0) * (1.0 - mean0)) * count0) / pix) * (pix / (pix - 1)));
//	sd1 = std::sqrtf(((((1.0 - mean1) * (1.0 - mean1)) * count1) / pix) * (pix / (pix - 1)));
//	Population.copyTo(StartPopulation);
//
//	return true;
//}
//
//bool indicator_kriging::majorityFilter()
//{
//	cv::Mat tmp;
//	Population.copyTo(tmp);
//	for (int i = 0; i < CurrentImages.rows; ++i)
//	{
//		for (int j = 0; j < CurrentImages.cols; ++j)
//		{
//			if (StartPopulation.at<uchar>(i, j) == Population0 || StartPopulation.at<uchar>(i, j) == Population1)
//				runKernelMF(i, j, tmp);
//		}
//	}
//
//	cv::imshow("img", CurrentImages);
//	cv::imshow("tresh", tmp);
//	cv::waitKey(0);
//	cv::imshow("MF", Population);
//	cv::imshow("ind0", indicator0);
//	cv::imshow("ind1", indicator1);
//	//cv::waitKey(0);
//
//	return true;
//}
//
//void indicator_kriging::runKernelMF(int r, int c, const cv::Mat tempPopulation)
//{
//	int countPixWin = 0;
//	int value = StartPopulation.at<uchar>(r, c);
//
//	int P0 = 0;
//	int P1 = 0;
//
//	for (int i = r - radiusKernelMF; i <= r + radiusKernelMF; ++i)
//	{
//		for (int j = c - radiusKernelMF; j <= c + radiusKernelMF; ++j)
//		{
//			if (i >= 0 && j >= 0 && i < Population.rows && j < Population.cols)
//			{
//				++countPixWin;
//				if (tempPopulation.at<uchar>(i, j) == Population0) // TODO it must run only for start population, but on second call its not
//					++P0;
//				else //if (StartPopulation.at<uchar>(i, j) == Population1)
//					++P1;
//			}
//		}
//	}
//
//	if (1.0*P0 / countPixWin > majorityTresh && value == Population1)
//	{
//		//indicator0.at<float>(r, c) = 1.0;
//		//indicator1.at<float>(r, c) = 1.0;
//		Population.at<uchar>(r, c) = Population0;
//	}
//	else if (1.0*P1 / countPixWin > majorityTresh && value == Population0)
//	{
//		//indicator0.at<float>(r, c) = 0.0;
//		//indicator1.at<float>(r, c) = 0.0;
//		Population.at<uchar>(r, c) = Population1;
//	}
//}
//
//bool indicator_kriging::calcIndicator()
//{
//	float sl0 = 0;
//	float sr1 = 0;
//	//float sr0 = (sd0 * T1 + sd1 * T0) / (sd0 + sd1);
//	float x = (sd0 * T1 + sd1 * T0) / (sd0 + sd1);
//	float sr0 = x - T0;
//	float sl1 = T1 - x;
//
//	float Tsl0 = T0 - sl0;
//	float Tsr0 = T0 + sr0;
//	float Tsl1 = T1 - sl1;
//	float Tsr1 = T1 + sr1;
//
//	indicator0.create(CurrentImages.rows, CurrentImages.cols, CV_32FC1);
//	indicator1.create(CurrentImages.rows, CurrentImages.cols, CV_32FC1);
//	for (int i = 0; i < CurrentImages.rows; ++i)
//	{
//		for (int j = 0; j < CurrentImages.cols; ++j)
//		{
//			//t0 if x < T: ind = 1
//			if (CurrentImages.at<uchar>(i, j) < Tsl0)
//				indicator0.at<float>(i, j) = 1.0;
//			else if (CurrentImages.at<uchar>(i, j) > Tsr0)
//				indicator0.at<float>(i, j) = 0.0;
//			else
//			{
//				float tmp = cumulative(Tsr0);
//				float res = (tmp - cumulative(CurrentImages.at<uchar>(i, j))) / (tmp - cumulative(Tsl0));
//				res = (res > 1.0) ? 1.0 : res;
//				indicator0.at<float>(i, j) = (res < 0.0) ? 0.0 : res;
//			}
//			//t1
//			if (CurrentImages.at<uchar>(i, j) < Tsl1)
//				indicator1.at<float>(i, j) = 1.0;
//			else if (CurrentImages.at<uchar>(i, j) > Tsr1)
//				indicator1.at<float>(i, j) = 0.0;
//			else
//			{
//				float tmp = cumulative(Tsr1);
//				float res = (tmp - cumulative(CurrentImages.at<uchar>(i, j))) / (tmp - cumulative(Tsl1));
//				res = (res > 1.0) ? 1.0 : res;
//				indicator1.at<float>(i, j) = (res < 0.0) ? 0.0 : res;
//			}
//		}
//	}
//	return true;
//}
//
//float indicator_kriging::cumulative(float x)
//{
//	int idx = (x + 0.5 > 255) ? 255 : (x + 0.5);
//	idx = (idx < 0.0) ? 0 : idx;
//	return probHist[idx];
//}
//
//bool indicator_kriging::calcCovariance()
//{
//	std::vector<float> seqCov0[29];
//	std::vector<float> seqCov1[29];
//	float sum0[29];
//	float sum1[29];
//	memset(sum0, 0, sizeof(float)* 29);
//	memset(sum1, 0, sizeof(float)* 29);
//	int N = CurrentImages.rows * CurrentImages.cols;
//	//for (int i = 0; i < 28; ++i)
//	//{
//	//	seqCov0[i].reserve(N);
//	//	seqCov1[i].reserve(N);
//	//}
//	int vidx = 0;
//
//	for (int r = 0; r < CurrentImages.rows; ++r)
//	{
//		for (int c = 0; c < CurrentImages.cols; ++c)
//		{
//			int idx = 0;
//			for (int i = r - radiusF; i <= r + radiusF; ++i)
//			{
//				for (int j = c - radiusF; j <= c + radiusF; ++j)
//				{
//					if (i >= 0 && j >= 0 && i < CurrentImages.rows && j < CurrentImages.cols)
//					{
//						seqCov0[idx].push_back(indicator0.at<float>(i, j));
//						seqCov1[idx].push_back(indicator1.at<float>(i, j));
//					}
//					else
//					{
//						seqCov0[idx].push_back(0.5);
//						seqCov1[idx].push_back(0.5);
//					}
//					sum0[idx] += seqCov0[idx][vidx];
//					sum1[idx] += seqCov1[idx][vidx];
//					++idx;
//				}
//			}
//			{ // 4 pixel
//				int i = r - radiusF;
//				int j = c;
//				if (i > 0)
//				{
//					seqCov0[idx].push_back(indicator0.at<float>(i, j));
//					seqCov1[idx].push_back(indicator1.at<float>(i, j));
//				}
//				else {
//					seqCov0[idx].push_back(0.5);
//					seqCov1[idx].push_back(0.5);
//				}
//				sum0[idx] += seqCov0[idx][vidx];
//				sum1[idx] += seqCov1[idx][vidx];
//				++idx; //////
//				i = r + radiusF;
//				j = c;
//				if (i < CurrentImages.rows)
//				{
//					seqCov0[idx].push_back(indicator0.at<float>(i, j));
//					seqCov1[idx].push_back(indicator1.at<float>(i, j));
//				}
//				else {
//					seqCov0[idx].push_back(0.5);
//					seqCov1[idx].push_back(0.5);
//				}
//				sum0[idx] += seqCov0[idx][vidx];
//				sum1[idx] += seqCov1[idx][vidx];
//				++idx; /////
//				i = r;
//				j = c - radiusF;
//				if (j > 0)
//				{
//					seqCov0[idx].push_back(indicator0.at<float>(i, j));
//					seqCov1[idx].push_back(indicator1.at<float>(i, j));
//				}
//				else {
//					seqCov0[idx].push_back(0.5);
//					seqCov1[idx].push_back(0.5);
//				}
//				sum0[idx] += seqCov0[idx][vidx];
//				sum1[idx] += seqCov1[idx][vidx];
//				++idx; ////////
//				i = r;
//				j = c + radiusF;
//				if (j < CurrentImages.cols)
//				{
//					seqCov0[idx].push_back(indicator0.at<float>(i, j));
//					seqCov1[idx].push_back(indicator1.at<float>(i, j));
//				}
//				else {
//					seqCov0[idx].push_back(0.5);
//					seqCov1[idx].push_back(0.5);
//				}
//				sum0[idx] += seqCov0[idx][vidx];
//				sum1[idx] += seqCov1[idx][vidx];
//			}
//			++vidx;
//		}
//	}
//
//	int n = 28;
//	cv::Mat a0(n + 1, n + 1, CV_32FC1);
//	cv::Mat y0(n + 1, 1, CV_32FC1);
//	x0.zeros(n + 1, 1, CV_32FC1);
//
//	for (int i = 0; i < n; ++i)
//	{
//		for (int j = i; j < n; ++j)
//		{
//			if (j >= 12)
//				a0.at<float>(i, j) = covariance(seqCov0[i + 1], sum0[i + 1], seqCov0[j + 1], sum0[j + 1]);
//			else
//				a0.at<float>(i, j) = covariance(seqCov0[i], sum0[i], seqCov0[j], sum0[j]);
//			a0.at<float>(j, i) = a0.at<float>(i, j);
//		}
//	}
//	for (int i = 0; i < n; ++i)
//	{
//		a0.at<float>(i, n) = 1.0;
//		a0.at<float>(n, i) = 1.0;
//	}
//	a0.at<float>(n, n) = 0.0;
//	for (int i = 0; i < n; ++i)
//	{
//		if (i >= 12)
//			y0.at<float>(i, 0) = covariance(seqCov0[i + 1], sum0[i + 1], seqCov0[12], sum0[12]);
//		else
//			y0.at<float>(i, 0) = covariance(seqCov0[i], sum0[i], seqCov0[12], sum0[12]);
//	}
//	y0.at<float>(n, 0) = 1.0;
//	cv::solve(a0, y0, x0, cv::DECOMP_SVD);
//	//std::cout << "X0 befor:" << std::endl;
//	//for (int i = 0; i < n; ++i)
//	//	std::cout << x0.at<float>(i, 0) << std::endl; 
//
//	normolize(x0, a0);
//
//	cv::Mat a1(n + 1, n + 1, CV_32FC1);
//	cv::Mat y1(n + 1, 1, CV_32FC1);
//	x1.zeros(n + 1, 1, CV_32FC1);
//	for (int i = 0; i < n; ++i)
//	{
//		for (int j = i; j < n; ++j)
//		{
//			if (j >= 12)
//				a1.at<float>(i, j) = covariance(seqCov1[i + 1], sum1[i + 1], seqCov1[j + 1], sum1[j + 1]);
//			else
//				a1.at<float>(i, j) = covariance(seqCov1[i], sum1[i], seqCov1[j], sum1[j]);
//			a1.at<float>(j, i) = a1.at<float>(i, j);
//		}
//	}
//	for (int i = 0; i < n; ++i)
//	{
//		a1.at<float>(i, n) = 1.0;
//		a1.at<float>(n, i) = 1.0;
//	}
//	a1.at<float>(n, n) = 0.0;
//	for (int i = 0; i < n; ++i)
//	{
//		if (i >= 12)
//			y1.at<float>(i, 0) = covariance(seqCov1[i + 1], sum1[i + 1], seqCov1[12], sum1[12]);
//		else
//			y1.at<float>(i, 0) = covariance(seqCov1[i], sum1[i], seqCov1[12], sum1[12]);
//	}
//	y1.at<float>(n, 0) = 1.0;
//	cv::solve(a1, y1, x1, cv::DECOMP_SVD);
//
//	std::cout << "X1 befor:" << std::endl;
//	for (int i = 0; i < n; ++i)
//		std::cout << x1.at<float>(i, 0) << std::endl;
//	normolize(x1, a1);
//
//	//std::cout << "X0 after2:" << std::endl;
//	//for (int i = 0; i < n; ++i)
//	//	std::cout << x0.at<float>(i, 0) << std::endl; 
//	//std::cout << "X1 after2:" << std::endl;
//	//for (int i = 0; i < n; ++i)
//	//	std::cout << x1.at<float>(i, 0) << std::endl;
//	return true;
//}
////
////void indicator_kriging::normolize(cv::Mat& x, const cv::Mat a)
//{
//	float avgl = 0.0;
//	int n = x.rows;
//	cv::Mat AvgCov(n, 1, CV_32FC1, cv::Scalar(0.0));
//
//	int countNegativeWeights = 0;
//	for (int i = 0; i < n; ++i)
//	{
//		if (x.at<float>(i, 0) < 0.0)
//		{
//			avgl += x.at<float>(i, 0);
//			++countNegativeWeights;
//			for (int j = 0; j < n; ++j)
//			{
//				if (x.at<float>(j, 0) < 0.0)
//					AvgCov.at<float>(i, 0) += a.at<float>(i, j);
//			}
//		}
//	}
//	AvgCov = AvgCov / countNegativeWeights;
//	avgl = avgl / countNegativeWeights;
//
//	//std::cout << "AvgCov:" << std::endl;
//	//for (int i = 0; i < n; ++i)
//	//	std::cout << AvgCov.at<float>(i, 0) << std::endl;
//
//	float nSum = 0.0;
//	for (int i = 0; i < n; ++i)
//	{
//		if (x.at<float>(i, 0) < 0.0)
//		{
//			x.at<float>(i, 0) = 0.0;
//			if (x.at<float>(i, 0) < std::abs(avgl) && AvgCov.at<float>(i, 0) < a.at<float>(i, 12))
//				++countNegativeWeights;
//		}
//		nSum += x.at<float>(i, 0);
//	}
//	x = x / nSum;
//}
//
//double indicator_kriging::covariance(std::vector<float> a, float suma, std::vector<float> b, float sumb)
//
//	double mult = (suma / a.size()) * (sumb / b.size());
//	double sum = 0;
//	for (int i = 0; i < a.size(); ++i)
//	{
//		sum += a[i] * b[i];
//	}
//	sum = sum / a.size();
//	return (sum - mult);
//
///
///void indicator_kriging::calcProbability(cv::Mat& prob, const cv::Mat kernel, int T)
//
//	int idx = 0;
//	cv::Mat Ker(2 * radiusF + 2, 2 * radiusF + 2, CV_32FC1, cv::Scalar(0.0));
//	for (int r = 1; r < Ker.rows - 1; ++r)
//	{
//		for (int c = 1; c < Ker.rows - 1; ++c)
//		{
//			if (!(c == radiusF + 1 && r == radiusF + 1))
//				Ker.at<float>(r, c) = kernel.at<float>(idx++, 0);
//		}
//	}
//	cv::Mat Ind;
//	cv::threshold(CurrentImages, Ind, T, 1.0, cv::THRESH_BINARY_INV);
//
//	cv::filter2D(Ind, prob, CV_32FC1, Ker);
//
///
///void indicator_kriging::calcPrRun()
//
//	calcProbability(Probability0, x0, T0);
//	calcProbability(Probability1, x1, T1);
//	cv::imshow("prob0", Probability0);
//	cv::imshow("prob1", Probability1);
//
///
///void indicator_kriging::setPopulation()
//
//	for (int r = 0; r < CurrentImages.rows; ++r)
//	{
//		for (int c = 0; c < CurrentImages.cols; ++c)
//		{
//			if (Population.at<uchar>(r, c) == UnknowPopulation)
//			{
//				if (Probability0.at<float>(r, c) > 1 - Probability1.at<float>(r, c))
//					Population.at<uchar>(r, c) = Population0;
//				else
//					Population.at<uchar>(r, c) = Population1;
//			}
//		}
//	}
//
//	majorityFilter();
//
//	cv::imshow("population_result", Population);
//	cv::waitKey(0);
//
//
//void indicator_kriging::save(char* fname)
//{
//	cv::imwrite(fname, Population);
//}