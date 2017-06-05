#include "kriging.h"


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

void kriging::show() const
{
	if (m_inputImg.data)
	{
		cv::imshow("Input", m_inputImg);
		if (m_threshold.data)
			cv::imshow("Output", m_threshold);
		cv::waitKey(0);
	}
}

bool kriging::setT(unsigned char t0, unsigned char t1)
{
	m_T1 = std::max(t0, t1);
	m_T0 = std::min(t0, t1);
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

	/*//////////////////
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	// Normalize the result to [ 0, histImage.rows ]
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	cv::imshow("calcHist Demo", histImage);

	cv::waitKey(0);
	///////////////////////////////
	*/



	m_cumProbHist = cv::Mat(hist.rows, hist.cols, CV_32FC1);
	m_probHist = cv::Mat(hist.rows, hist.cols, CV_32FC1);
	float sum = 0.0;
	for (int i = 0; i < m_cumProbHist.rows; ++i)
	{
		float prob = (hist.at<float>(i, 0) / m_numAllPixels);
		m_cumProbHist.at<float>(i, 0) = prob + sum;
		m_probHist.at<float>(i, 0) = prob;
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

	cv::imshow("thres", m_threshold);
	cv::waitKey(0);
	return true;
}

bool kriging::calcIndicator()
{
	if (!m_inputImg.data)
		return false;

	float sl0 = 0;
	float sr1 = 0;
	float x = 0.0;
	if (!(m_sd0 == 0.0 && m_sd1 == 0.0))
		x = (m_sd0 * m_T1 + m_sd1 * m_T0) / (m_sd0 + m_sd1);
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


//float kriging::calcEtrophy() 
//{
//	float Entrophy = 0.0;
//	for (int i = 0; i < m_probHist.rows; ++i)
//	{
//		float value = m_probHist.at<float>(i, 0);
//		if (value != 0.0)
//			Entrophy += value * std::log(value);
//	}
//	Entrophy = -Entrophy;
//}

const float treshProb = 0.005;

float kriging::calcKapurEtrophy(int T)
{
	if (T < 0 || T > 255)
		return false;

	float probabilityLeft =  m_cumProbHist.at<float>(T, 0);
	if (probabilityLeft <= treshProb || probabilityLeft >= 1.0 - treshProb)
		return 0.0;

	float probabilityRight = 1 - probabilityLeft;

	float entrophyLeft = 0.0;
	float Entrophy = 0.0;

	int i = 0;
	for (i = 0; i <= T; ++i)
	{
		float value = m_probHist.at<float>(i, 0);
		if (value != 0.0)
			entrophyLeft += value * std::log(value);
	}
	for (; i <= 255; ++i)
	{
		float value = m_probHist.at<float>(i, 0);
		if (value != 0.0)
			Entrophy += value * std::log(value);
	}

	Entrophy = -(Entrophy + entrophyLeft);
	entrophyLeft = -entrophyLeft;
	float a = std::log(probabilityLeft*probabilityRight);
	float b = (entrophyLeft / probabilityLeft);
	float c = ((Entrophy - entrophyLeft) / probabilityRight);

	return std::log(probabilityLeft*probabilityRight) + (entrophyLeft / probabilityLeft) + ((Entrophy - entrophyLeft) / probabilityRight);
}

void kriging::chooseThreshold(float r)
{
	r = (r > 1.0) ? 1.0 : r;
	r = (r < 0.0) ? 0.0 : r;
	float entrophy[256];
	float max = 0.0;
	int maxIdx;
	for (int i = 0; i <= 255; ++i)
	{
		entrophy[i] = calcKapurEtrophy(i);
		if (max <= entrophy[i])
		{
			max = entrophy[i];
			maxIdx = i;
		}
	}

	float inv_r = 1.0 - r;
	float fi = inv_r * max;
	for (int i = maxIdx - 1; i != 0; --i)
	{
		if (entrophy[i] <= fi)
		{
			m_T0 = i;
			break;
		}
	}
	for (int i = maxIdx + 1; i <= 255; ++i)
	{
		if (entrophy[i] <= fi)
		{
			m_T1 = i;
			break;
		}
	}
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

cv::Mat fixedWindowKriging::getKrigingKernel(const cv::Mat& weightsMatrix)
{
	int kernelSize = 2 * m_radiusKrigng + 1;
	cv::Mat krigingKernel(kernelSize, kernelSize, CV_32FC1, cv::Scalar(0.0));
	int size_system = m_numElemUnderWindow - 1;
	for (int i = 0; i < size_system; ++i)
		krigingKernel.at<float>(m_krigingKernelIndex[i].first + m_radiusKrigng, m_krigingKernelIndex[i].second + m_radiusKrigng) = weightsMatrix.at<float>(i, 0);

	return krigingKernel;
}


void fixedWindowKriging::write(const cv::String& imgName, bool test)
{
	m_indicator0.convertTo(m_indicator0, CV_8UC1, 255.0);
	m_indicator1.convertTo(m_indicator1, CV_8UC1, 255.0);
	m_probabilityPopulation0.convertTo(m_probabilityPopulation0, CV_8UC1, 255.0);
	m_probabilityPopulation1.convertTo(m_probabilityPopulation1, CV_8UC1, 255.0);

	cv::String path;
	if (test)
		path = "..//images//output//tests//" + imgName;
	else 
		path = "..//images//output//" + imgName;

	cv::String thresh = cv::format("_%i_%i.", m_T0, m_T1);
	cv::String a = path + "_initial_threshold" + thresh + "png";
	if (m_initialPopulation.data)
		cv::imwrite(path + "_initial_threshold" + thresh + "png", m_initialPopulation);
	if (m_indicator0.data)
		cv::imwrite(path + "_indicator0" + thresh + "png", m_indicator0);
	if (m_indicator1.data)
		cv::imwrite(path + "_indicator1" + thresh + "png", m_indicator1);
	if (m_probabilityPopulation0.data)
		cv::imwrite(path + "_probability_population0" + thresh + "png", m_probabilityPopulation0);
	if (m_probabilityPopulation1.data)
		cv::imwrite(path + "_probability_population1" + thresh + "png", m_probabilityPopulation1);
	if (m_inputImg.data)
		cv::imwrite(path + "_Init" + thresh + "png", m_inputImg);
	if (m_threshold.data)
		cv::imwrite(path + " _output" + thresh + "png", m_threshold);

	std::cout.width(5);

	std::ofstream file0;
	std::ofstream file1;
	file0.open(path + "_krigingKernel_population0" + thresh + "txt");
	file1.open(path + "_krigingKernel_population1" + thresh + "txt");
	int kernelSize = 2 * m_radiusKrigng + 1;
	for (int row = 0; row < kernelSize; ++row)
	{
		for (int col = 0; col < kernelSize; ++col)
		{
			file0 << std::setw(10) << m_krigingKernel0.at<float>(row, col);
			file1 << std::setw(10) << m_krigingKernel1.at<float>(row, col);
		}
		file0 << std::endl;
		file1 << std::endl;
	}
	file0.close();
	file1.close();

	std::ofstream f0, f1;
	f0.open(path + "_krigingSystem_population0" + thresh + "txt");
	f1.open(path + "_krigingSystem_population1" + thresh + "txt");
	for (int row = 0; row < m_numElemUnderWindow; ++row)
	{
		for (int col = 0; col < m_numElemUnderWindow; ++col)
		{
			f0 << std::setw(10) << m_krigingSystemLeft_0.at<float>(row, col);
			f1 << std::setw(10) << m_krigingSystemLeft_1.at<float>(row, col);
		}
		f0 << " = " << m_krigingSystemRight_0.at<float>(row, 0) << std::endl;
		f1 << " = " << m_krigingSystemRight_1.at<float>(row, 0) << std::endl;
	}
	f0.close();
	f1.close();
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

	cv::Mat sequences0(m_numAllPixels, m_numElemUnderWindow, CV_32FC1);
	cv::Mat sequences1(m_numAllPixels, m_numElemUnderWindow, CV_32FC1);
	int counter = 0;
	for (int row = m_radiusKrigng; row < indicator0.rows - m_radiusKrigng; ++row)
	{
		for (int col = m_radiusKrigng; col < indicator0.cols - m_radiusKrigng; ++col)
		{
			int i = 0;
			for (const auto coord : m_krigingKernelIndex)
			{
				sequences0.at<float>(counter, i) = indicator0.at<float>(row + coord.first, col + coord.second);
				sequences1.at<float>(counter, i++) = indicator1.at<float>(row + coord.first, col + coord.second);
			}
			++counter;
		}
	}

	m_krigingSystemLeft_0 = getKrigingSystem(sequences0, false);
	m_krigingSystemLeft_1 = getKrigingSystem(sequences1, false);
	m_krigingSystemRight_0 = getKrigingSystem(sequences0, true);
	m_krigingSystemRight_1 = getKrigingSystem(sequences1, true);

	cv::Mat solveKrigingSystem_0;
	cv::Mat solveKrigingSystem_1;

	if (!cv::solve(m_krigingSystemLeft_0, m_krigingSystemRight_0, solveKrigingSystem_0, cv::DECOMP_SVD))
		return false;

	if (!cv::solve(m_krigingSystemLeft_1, m_krigingSystemRight_1, solveKrigingSystem_1, cv::DECOMP_SVD))
		return false;

	escapeNegativeWeights(solveKrigingSystem_0, m_krigingSystemRight_0);
	escapeNegativeWeights(solveKrigingSystem_1, m_krigingSystemRight_1);

	m_krigingKernel0 = getKrigingKernel(solveKrigingSystem_0);
	m_krigingKernel1 = getKrigingKernel(solveKrigingSystem_1);

	return true;
}

bool fixedWindowKriging::calcProbability()
{
	if (!m_krigingKernel0.data || !m_krigingKernel1.data)
		return false;

	cv::filter2D(m_indicator0, m_probabilityPopulation0, CV_32FC1, m_krigingKernel0, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::filter2D(m_indicator1, m_probabilityPopulation1, CV_32FC1, m_krigingKernel1);

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

	return true;
}
