#include "kriging.h"

#define Population0 10
#define Population1 254
#define UnknowPopulation 125

kriging::kriging(int radiusMF, float threshMF) : m_radiusMF(radiusMF), m_threshMF(threshMF)
{
	m_T0 = 0;
	m_T1 = 0;
}

bool kriging::read(const cv::String& fname)
{
	m_inputImg = cv::imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	if (!m_inputImg.data)
		return false;
	m_numAllPixels = m_inputImg.total();
	if (m_numAllPixels < 1)
		return false;
	return true;
}

bool kriging::calcHist()
{
	if (!m_inputImg.data)
		return false;

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::Mat hist;

	cv::calcHist(&m_inputImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
	m_cumProbHist = cv::Mat(hist.rows, hist.cols, CV_32FC1);
	float sum = 0.0;
	for (int i = 0; i < m_cumProbHist.rows; ++i)
	{
		float prob = (hist.at<float>(i, 0) / m_numAllPixels);
		m_cumProbHist.at<float>(i, 0) = prob + sum;
		sum += prob;
	}	
	return true;
}

bool kriging::thresholding()
{
	if (!m_inputImg.data || (m_T0 == 0 && m_T1 == 0))
		return false;

	float mean0 = 0.0;
	float mean1 = 0.0;

	m_threshold = cv::Mat(m_inputImg.rows, m_inputImg.cols, CV_8UC1);
	if (!m_threshold.data)
		return false;

	unsigned char* ptr_inputImg = m_inputImg.data;
	unsigned char* ptr_threshold = m_threshold.data;
	for (int i = 0; i < m_numAllPixels; ++i)
	{
		if (m_T0 > *ptr_inputImg)
		{
			*ptr_threshold++ = Population0;
			++mean0;
		}
		else if (m_T1 < *ptr_inputImg)
		{
			*ptr_threshold++ = Population1;
			++mean1;
		}
		else 
			*ptr_threshold++ = UnknowPopulation;
		++ptr_inputImg;
	}

	int count0 = mean0;
	int count1 = mean1;
	mean0 = mean0 / m_numAllPixels;
	mean1 = mean1 / m_numAllPixels;
	m_sd0 = std::sqrtf((((1.0 - mean0) * (1.0 - mean0)) * count0) / m_numAllPixels);
	m_sd1 = std::sqrtf((((1.0 - mean1) * (1.0 - mean1)) * count1) / m_numAllPixels);

	m_threshold.copyTo(m_initialPopulation);
	//cv::threshold(m_inputImg, m_threshold, m_T0, 0, CV_THRESH_TOZERO);
	//cv::threshold(m_threshold, m_threshold, m_T1, 0, CV_THRESH_TRUNC);
	return true;
}

bool kriging::calcIndicator()
{
	if (!m_inputImg.data)
		return false;

	float sl0 = 0;
	float sr1 = 0;
	float x = (m_sd0 * m_T1 + m_sd1 * m_T0) / (m_sd0 + m_sd1);
	float sr0 = x - m_T0;
	float sl1 = m_T1 - x;
	
	float Tsl0 = m_T0 - sl0;
	float Tsr0 = m_T0 + sr0;
	float Tsl1 = m_T1 - sl1;
	float Tsr1 = m_T1 + sr1;

	m_indicator0.create(m_inputImg.rows, m_inputImg.cols, CV_32FC1);
	m_indicator1.create(m_inputImg.rows, m_inputImg.cols, CV_32FC1);
	if (!m_indicator0.data || !m_indicator1.data)
		return false;

	for (int row = 0; row < m_inputImg.rows; ++row)
	{
		for (int col = 0; col < m_inputImg.cols; ++col)
		{
			//if x < T0: ind0 = 1.0
			if (m_inputImg.at<uchar>(row, col) < Tsl0)
				m_indicator0.at<float>(row, col) = 1.0;
			else if (m_inputImg.at<uchar>(row, col) > Tsr0)
				m_indicator0.at<float>(row, col) = 0.0;
			else
			{
				float temp = m_cumProbHist.at<float>(Tsr0, 0);
				m_indicator0.at<float>(row, col) = 
					(temp - m_cumProbHist.at<float>(m_inputImg.at<uchar>(row, col), 0)) / (temp - m_cumProbHist.at<float>(Tsl0, 0));
			}
			// if x < T1: ind1 = 1.0
			if (m_inputImg.at<uchar>(row, col) < Tsl1)
				m_indicator1.at<float>(row, col) = 1.0;
			else if (m_inputImg.at<uchar>(row, col) > Tsr1)
				m_indicator1.at<float>(row, col) = 0.0;
			else
			{
				float temp = m_cumProbHist.at<float>(Tsr1, 0);
				m_indicator1.at<float>(row, col) = 
					(temp - m_cumProbHist.at<float>(m_inputImg.at<uchar>(row, col), 0)) / (temp - m_cumProbHist.at<float>(Tsl1, 0));
			}
		}
	}
	return true;
}

bool kriging::setT(unsigned char t0, unsigned char t1)
{
	m_T1 = std::max(t0, t1);
	m_T0 = std::min(t0, t1);
	return true;
}

bool kriging::majorityFilter()
{
	if (!m_threshold.data || !m_initialPopulation.data)
		return false;

	cv::Mat temp_thresh;
	m_threshold.copyTo(temp_thresh);

	float majortyThresh = (m_radiusMF * 2. + 1.) * (m_radiusMF * 2. + 1.) * m_threshMF;
	int skip = m_radiusMF * m_threshold.cols;
	unsigned char* ptr_initialPopualtion = m_initialPopulation.data + skip;
	unsigned char* ptr_threshold = m_threshold.data + skip;
	unsigned char* ptr_temp_thresh = temp_thresh.data + skip;
	int limit = m_numAllPixels - skip;
	int k = 0;
	for (int row = m_radiusMF; row < m_threshold.rows - m_radiusMF; ++row)
	{
		for (int j = 0; j < m_radiusMF; ++j)
		{
			++ptr_initialPopualtion;
			++ptr_threshold;
			++ptr_temp_thresh;
		}
		for (int col = m_radiusMF; col < m_threshold.cols - m_radiusMF; ++col)
		{
			if (*ptr_initialPopualtion != UnknowPopulation)
			{
				int countP0 = 0;
				int countP1 = 0;
				for (int rowKernel = -m_radiusMF; rowKernel <= m_radiusMF; ++rowKernel)
				{
					skip = rowKernel * m_initialPopulation.cols;
					for (int colKernel = -m_radiusMF; colKernel <= m_radiusMF; ++colKernel)
					{
						unsigned char value = *(ptr_temp_thresh + skip + colKernel);
						if (value == Population0)
							++countP0;
						else if (value == Population1)
							++countP1;
					}
				}
				unsigned char value = *ptr_temp_thresh;
				if (countP0 > majortyThresh && value != Population0)
				{
					int skip_ind = skip + (row * m_radiusMF) + col;
					*(m_indicator0.data + skip_ind) = 1.0;
					*(m_indicator1.data + skip_ind) = 1.0;
					*ptr_threshold = Population0;
				}
				else if (countP1 > majortyThresh && value != Population1)
				{
					int skip_ind = skip + (row * m_radiusMF) + col;
					*(m_indicator0.data + skip_ind) = 0.0;
					*(m_indicator1.data + skip_ind) = 0.0;
					*ptr_threshold = Population1;
				}
			}
			++ptr_initialPopualtion;
			++ptr_threshold;
			++ptr_temp_thresh;
		}
		for (int j = 0; j < m_radiusMF; ++j)
		{
			++ptr_initialPopualtion;
			++ptr_threshold;
			++ptr_temp_thresh;
		}
	}

	return true;
}

void kriging::escapeNegativeWeights(cv::Mat& weightMatrix, const cv::Mat& krigingSystemRight) const
{
	float avgl = 0.0;
	float avgc = 0.0;

	int system_size = weightMatrix.rows - 1;
	cv::Mat AvgCov(system_size, 1, CV_32FC1, cv::Scalar(0.0));
	
	int countNegativeWeights = 0;
	for (int row = 0; row < system_size; ++row)
	{
		if (weightMatrix.at<float>(row, 0) < 0.0)
		{
			avgl += weightMatrix.at<float>(row, 0);
			avgc += krigingSystemRight.at<float>(row, 0);
			++countNegativeWeights;
		}
	}
	avgc = avgc / countNegativeWeights;
	avgl = avgl / countNegativeWeights;
	
	float limit_avg_l = std::abs(avgl);
	float nSum = 0.0;
	for (int row = 0; row < system_size; ++row)
	{
		if (weightMatrix.at<float>(row, 0) < 0.0)
			weightMatrix.at<float>(row, 0) = 0.0;
		else if (weightMatrix.at<float>(row, 0) < limit_avg_l && krigingSystemRight.at<float>(row, 0) < avgc)
			weightMatrix.at<float>(row, 0) = 0.0;
		nSum += weightMatrix.at<float>(row, 0);
	}
	weightMatrix = weightMatrix / nSum;
}

void fixedWindowKriging::write(const cv::String& imgName)
{
	const cv::String path = "..//images//output//" + imgName;
	cv::String thresh = cv::format("_%i_%i.", m_T0, m_T1);
	cv::String a = path + "_initial_threshold" + thresh + "png";
	cv::imwrite(path + "_initial_threshold" + thresh + "png", m_initialPopulation);
	cv::imwrite(path + "_indicator0" + thresh + "png", m_indicator0);
	cv::imwrite(path + "_indicator1" + thresh + "png", m_indicator1);
	cv::imwrite(path + "_probability_population0" + thresh + "png", m_probabilityPopulation0);
	cv::imwrite(path + "_probability_population1" + thresh + "png", m_probabilityPopulation1);
	cv::imwrite(path + "_Init" + thresh + "png", m_inputImg);
	cv::imwrite(path + " _output" + thresh + "png", m_threshold);

	std::ofstream file0;
	std::ofstream file1;
	file0.open("..//images//output//" + imgName + "_krigingKernel_population0" + thresh + "txt");
	file1.open("..//images//output//" + imgName + "_krigingKernel_population1" + thresh + "txt");
	file0 << m_krigingKernel0;
	file1 << m_krigingKernel1;
	file0.close();
	file1.close();
}

fixedWindowKriging::fixedWindowKriging(int radiusKriging, int radiusMF, float threshMF) 
: kriging(radiusMF, threshMF), m_radiusKrigng(radiusKriging) {}

//void fixedWindowKriging::setKernelIndexArray()
//{
//	int col = 0;
//	for (int row = 0; row < m_radiusKrigng;)
//	{
//		if (row == col  && row != 0)
//		{
//			col = 0;
//			++row;
//		}
//		else if (row > col)
//			++col;
//		else
//			++row;
//
//		if (col == 0 || col == row)
//		{
//			m_krigingKernelIndex.push_back(std::pair<int, int>(-row, col));			// 1		1			  1 x 2
//			m_krigingKernelIndex.push_back(std::pair<int, int>(col, row));			// 2	  4 x 2		or	  x x x
//			m_krigingKernelIndex.push_back(std::pair<int, int>(row, -col));			// 3		3			  4 x 3
//			m_krigingKernelIndex.push_back(std::pair<int, int>(-col, -row));		// 4
//		}
//		else
//		{
//			m_krigingKernelIndex.push_back(std::pair<int, int>(-row, col));			// 1
//			m_krigingKernelIndex.push_back(std::pair<int, int>(-col, row));			// 2			   8 x 1
//			m_krigingKernelIndex.push_back(std::pair<int, int>(col, row));			// 3			 7 x x x 2
//			m_krigingKernelIndex.push_back(std::pair<int, int>(row, col));			// 4			 x x x x x
//			m_krigingKernelIndex.push_back(std::pair<int, int>(row, -col));			// 5			 6 x x x 3
//			m_krigingKernelIndex.push_back(std::pair<int, int>(col, -row));			// 6			   5 x 4
//			m_krigingKernelIndex.push_back(std::pair<int, int>(-col, -row));		// 7
//			m_krigingKernelIndex.push_back(std::pair<int, int>(-row, -col));		// 8
//		}
//	}
//}

void fixedWindowKriging::setKernelIndexArray()
{
	for (int row = -m_radiusKrigng; row <= m_radiusKrigng; ++row)
	{
		for (int col = -m_radiusKrigng; col <= m_radiusKrigng; ++col)							//			   00
		{																						//		 01 02 03 04 05																	//	  11 12 13 xx .. .. ..
			if (std::sqrt(row*row + col *col) <= m_radiusKrigng && !(row == 0 && col == 0))		//		 06 07 08 09 10
				m_krigingKernelIndex.push_back(std::pair<int, int>(row, col));					//	  11 12 13 28 14 15 16
		}																						//		 17 18 19 20 21
	}																							//		 22 23 24 25 26																									//			   27
	m_krigingKernelIndex.push_back(std::pair<int, int>(0, 0));									//			   27
	m_numElemUnderWindow = m_krigingKernelIndex.size();
}

bool fixedWindowKriging::calcCovarianceMatrix()
{
	if (!m_indicator0.data || !m_indicator1.data)
		return false;

	setKernelIndexArray();

	cv::Mat indicator0;
	cv::Mat indicator1;

	cv::copyMakeBorder(m_indicator0, indicator0, m_radiusKrigng, m_radiusKrigng, m_radiusKrigng, m_radiusKrigng, cv::BORDER_CONSTANT, cv::Scalar(0.5));
	cv::copyMakeBorder(m_indicator1, indicator1, m_radiusKrigng, m_radiusKrigng, m_radiusKrigng, m_radiusKrigng, cv::BORDER_CONSTANT, cv::Scalar(0.5));

	if ((indicator0.rows != indicator1.rows) || (indicator0.cols != indicator1.cols))
		return false;

	cv::Mat sequiences0(m_numAllPixels, m_numElemUnderWindow, CV_32FC1);
	cv::Mat sequiences1(m_numAllPixels, m_numElemUnderWindow, CV_32FC1);
	int counter = 0;
	for (int row = m_radiusKrigng; row < indicator0.rows - m_radiusKrigng; ++row)
	{
		for (int col = m_radiusKrigng; col < indicator0.cols - m_radiusKrigng; ++col)
		{
			int i = 0;
			for (const auto coord : m_krigingKernelIndex)
			{
				sequiences0.at<float>(counter, i) = indicator0.at<float>(row + coord.first, col + coord.second);
				sequiences1.at<float>(counter, i++) = indicator1.at<float>(row + coord.first, col + coord.second);
			}
			++counter;
		}
	}

	cv::Mat krigingSystemLeft_0 = getKrigingSystem(sequiences0, false);
	cv::Mat krigingSystemLeft_1 = getKrigingSystem(sequiences1, false);
	cv::Mat krigingSystemRight_0 = getKrigingSystem(sequiences0, true);
	cv::Mat krigingSystemRight_1 = getKrigingSystem(sequiences1, true);

	// write system to file
	cv::String thresh = cv::format("_%i_%i.", m_T0, m_T1);
	std::ofstream f0, f1;
	f0.open("..//images//output//krigingSystem_population0" + thresh + "txt");
	f1.open("..//images//output//krigingSystem_population1" + thresh + "txt");
	for (int row = 0; row < m_numElemUnderWindow; ++row)
	{
		for (int col = 0; col < m_numElemUnderWindow; ++col)
		{
			f0 << krigingSystemLeft_0.at<float>(row, col) << " ";
			f1 << krigingSystemLeft_1.at<float>(row, col) << " ";
		}
		f0 << " = " << krigingSystemRight_0.at<float>(row, 0) << std::endl;
		f1 << " = " << krigingSystemRight_1.at<float>(row, 0) << std::endl;
	}
	f0.close();
	f1.close();
	////////////////

	cv::Mat solveKrigingSystem_0;
	cv::Mat solveKrigingSystem_1;

	if (!cv::solve(krigingSystemLeft_0, krigingSystemRight_0, solveKrigingSystem_0, cv::DECOMP_SVD))
		return false;

	if (!cv::solve(krigingSystemLeft_1, krigingSystemRight_1, solveKrigingSystem_1, cv::DECOMP_SVD))
		return false;

	escapeNegativeWeights(solveKrigingSystem_0, krigingSystemRight_0);
	escapeNegativeWeights(solveKrigingSystem_1, krigingSystemRight_1);

	m_krigingKernel0 = getKrigignKernel(solveKrigingSystem_0);
	m_krigingKernel1 = getKrigignKernel(solveKrigingSystem_1);

	return true;
}

cv::Mat fixedWindowKriging::getKrigignKernel(const cv::Mat& weightsMatrix)
{
	int kernelSize = 2 * m_radiusKrigng + 1;
	cv::Mat krigingKernel(kernelSize, kernelSize, CV_32FC1, cv::Scalar(0.0));
	int size_system = m_numElemUnderWindow - 1;
	for (int i = 0; i < size_system; ++i)
		krigingKernel.at<float>(m_krigingKernelIndex[i].first + m_radiusKrigng, m_krigingKernelIndex[i].second + m_radiusKrigng) = weightsMatrix.at<float>(i, 0);

	return krigingKernel;
}


float fixedWindowKriging::covariance(const cv::Mat seq0, const cv::Mat seq1) const
{
	return (cv::sum(seq0.mul(seq1 / m_numAllPixels)) - (cv::mean(seq0) * cv::mean(seq1)))[0];
}

cv::Mat fixedWindowKriging::getKrigingSystem(const cv::Mat& sequencesMatrix, bool left_right) const
{
	if (!sequencesMatrix.data)
		return cv::Mat();

	if (m_krigingKernelIndex.empty())
		return cv::Mat();

	int system_size = m_numElemUnderWindow - 1;
	if (left_right)
	{
		cv::Mat krigingSystemRight(m_numElemUnderWindow, 1, CV_32FC1);
		for (int i = 0; i < system_size / 2; ++i)
		{
			krigingSystemRight.at<float>(i, 0) = covariance(sequencesMatrix.col(system_size), sequencesMatrix.col(i));
			krigingSystemRight.at<float>(system_size - 1 - i, 0) = krigingSystemRight.at<float>(i, 0);
		}
		krigingSystemRight.at<float>(system_size, 0) = 1.0;
		return krigingSystemRight;
	}
	else
	{
		cv::Mat krigingSystemLeft(m_numElemUnderWindow, m_numElemUnderWindow, CV_32FC1);
		for (int row = 0; row < system_size; ++row)
		{
			for (int col = row; col < system_size; ++col)
			{
				krigingSystemLeft.at<float>(row, col) = covariance(sequencesMatrix.col(row), sequencesMatrix.col(col));
				krigingSystemLeft.at<float>(col, row) = krigingSystemLeft.at<float>(row, col);
			}
		}
		cv::Mat(1, m_numElemUnderWindow, CV_32FC1, cv::Scalar(1.0)).copyTo(krigingSystemLeft.row(system_size));
		cv::Mat(m_numElemUnderWindow, 1, CV_32FC1, cv::Scalar(1.0)).copyTo(krigingSystemLeft.col(system_size));
		return krigingSystemLeft;
	}
}

bool fixedWindowKriging::calcProbability()
{
	if (!m_krigingKernel0.data || !m_krigingKernel1.data)
		return false;

	cv::imshow("ind0", m_indicator0);
	cv::imshow("ind1", m_indicator1);
	cv::imshow("initImg", m_inputImg);
	cv::imshow("initPop", m_initialPopulation);

	cv::filter2D(m_indicator0, m_probabilityPopulation0, CV_32FC1, m_krigingKernel0, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::filter2D(m_indicator1, m_probabilityPopulation1, CV_32FC1, m_krigingKernel1);

	cv::imshow("prob0", m_probabilityPopulation0);
	cv::imshow("prob1", m_probabilityPopulation1);
	//cv::waitKey(0);

	for (int row = 0; row < m_threshold.rows; ++row)
	{
		for (int col = 0; col < m_threshold.cols; ++col)
		{

			if (m_threshold.at<unsigned char>(row, col) == UnknowPopulation)
			{
				if (m_probabilityPopulation0.at<float>(row, col) > 1.0 - m_probabilityPopulation1.at<float>(row, col))
					m_threshold.at<unsigned char>(row, col) = Population0;
				else
					m_threshold.at<unsigned char>(row, col) = Population1;
			}
		}
	}
}

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