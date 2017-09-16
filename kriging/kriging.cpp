#include "kriging.h"
#include "string"

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
		{
			if (m_krigingSystemLeft_0.data)
				cv::imshow("Output kr", m_threshold);
			else
				cv::imshow("Output", m_threshold);
		}
		cv::waitKey(0);
	}
}

bool kriging::setT(unsigned char t0, unsigned char t1)
{
	m_T1 = std::max(t0, t1);
	m_T0 = std::min(t0, t1);
	return true;
}


void kriging::Kmeans(std::vector<double> t)
{
	int size_vector = t.size();
	int step = 256 / size_vector;

	int N = m_inputImg.total();
	unsigned char* data = m_inputImg.data;
	cv::Mat cluster(m_inputImg.size(), CV_8UC1);
	unsigned char* cl_data = cluster.data;

	std::vector<double> c(t);
	for (int j = 0;; ++j) 
	{//Запускаем основной цикл
		std::vector<int> sum(size_vector, 0);
		std::vector<int> n(size_vector, 0);
		std::vector<double> last_c(c);
		for (int i = 0; i < N; ++i)
		{
			double min = 513.0;
			int x = -1;
			for (int it = 0; it < size_vector; ++it)
			{
				if (min >= abs(c[it] - *data))
				{
					min = abs(c[it] - *data);
					x = it;
				}
			}
			sum[x] += *data;
			n[x] += 1;
			*cl_data = x * step;

			data++;
			cl_data++;
		}
		data = m_inputImg.data;
		cl_data = cluster.data;
		bool flag = true;

		for (int it = 0; it < size_vector; ++it)
		{
			c[it] = 1.0 * sum[it] / n[it];
			flag = (c[it] == last_c[it]);
		}

		for (int it = 0; it < size_vector; ++it)
			std::cout << c[it] << std::endl;
		
		std::cout << std::endl << std::endl;
		//std::cout << "iteration: " << j << std::endl;
		//std::cout << "c0: " << c0 << "    c1: " << c1 << "    c2: " << c2 << "    c3: " << c3 << std::endl;
		cv::imshow("img", m_inputImg);
		cv::imshow("cluster", cluster);
		cv::waitKey(0);


		if (flag) break;
	}

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::Mat hist;

	cv::calcHist(&m_inputImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

	//////////////////
	// Draw the histograms for B, G and R
	hist.copyTo(hist);
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
	for (auto it = c.begin(); it != c.end(); ++it)
	{
		line(histImage, cv::Point(bin_w*(*it), hist_h),
			cv::Point(bin_w*(*it), 100),
			cv::Scalar(0, 255, 0), 2, 8, 0);
	}

	/// Display
	cv::namedWindow("calcHist Demo (kMeans)", CV_WINDOW_AUTOSIZE);
	cv::imshow("calcHist Demo (kMeans)", histImage);
	cv::waitKey(0);
	///////////////////////////////

	m_T0 = c[0];
	m_T1 = c[1];
	std::cout << "c_T0: " << (int)m_T0 << std::endl;
	std::cout << "c_T1: " << (int)m_T1 << std::endl;

	/*cv::Mat tmp;
	for (int row = 0; row < m_inputImg.rows; ++row)
	{
		for (int col = 0; col < m_inputImg.cols; ++col)
		{
			if (m_inputImg.at<unsigned char>(row, col) > c[0] &&
				m_inputImg.at<unsigned char>(row, col) < c[1])
			{
				cv::Mat a(1, 1, CV_8UC1);
				a.at<unsigned char>(0, 0) = m_inputImg.at<unsigned char>(row, col);
				tmp.push_back(a);
			}
		}
	}
	tmp.copyTo(m_inputImg);*/
}

double kriging::calcmu(int s, int t)
{
	double sum = 0.0;
	for (int i = s; i < t; ++i)
		sum += m_probHist.at<float>(i, 0) * i;
	return sum;
}

void kriging::Otsu(int q)
{
	std::vector<int> t(q, 0);

	double max = -50.0;
	int _t0 = 0;
	int _t1 = 0;
	double muT = calcmu(0, 256);

	for (int t0 = 0; t0 < 256; ++t0)
	{
		for (int t1 = t0; t1 < 256; ++t1)
		{
			double w0 = m_cumProbHist.at<float>(t0, 0);
			double w1 = m_cumProbHist.at<float>(t1, 0) - w0;
			double w2 = 1.0 - m_cumProbHist.at<float>(t1, 0);
			double mu0 = calcmu(0, t0) / w0;
			double mu1 = calcmu(t0, t1) / w1;
			double mu2 = calcmu(t1, 256) / w2;
			double sigma0 = w0 * (mu0 - muT) * (mu0 - muT);
			double sigma1 = w1 * (mu1 - muT) * (mu1 - muT);
			double sigma2 = w2 * (mu2 - muT) * (mu2 - muT);
			double JJJ = sigma0 + sigma1 + sigma2;
			if (JJJ > max)
			{
				_t0 = t0;
				_t1 = t1;
				max = JJJ;
			}
		}
	}

	m_T0 = _t0;
	m_T1 = _t1;

	/////////////////////////////////////////////////////////
	std::vector<double> c;
	c.push_back(m_T0);
	c.push_back(m_T1);
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::Mat hist;

	cv::calcHist(&m_inputImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

	//////////////////
	// Draw the histograms for B, G and R
	hist.copyTo(hist);
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
	for (auto it = c.begin(); it != c.end(); ++it)
	{
		line(histImage, cv::Point(bin_w*(*it), hist_h),
			cv::Point(bin_w*(*it), 100),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	cv::imshow("calcHist Demo", histImage);
	cv::waitKey(0);
	///////////////////////////////
}

double NNN(double mu, double sigma, double x)
{
	double a = (1.0 / (sigma * std::sqrt(2.0 * M_PI)));
	double exp = std::exp(-(((x - mu) * (x - mu)) / (2 * sigma * sigma))); return a * exp;
}

void kriging::EM(int num_klass, float rb)
{
	double E = 0.1;
	double logSumPxn_prev = 0.0;
	int size = m_inputImg.rows * m_inputImg.cols;
	cv::Mat Pnk(size, num_klass, CV_64FC1);
	std::vector<double> w(num_klass, 1.0 / num_klass);
	std::vector<double> sigma(num_klass, 25.0);
	std::vector<double> mu(num_klass);
	for (int i = 1; i <= mu.size(); ++i)
		mu[i - 1] = 256.0 / (num_klass + 1) * i - 1.0;

	//a E
	std::vector<double> sumPxn(size, 0.0); 
	for (int clas = 0; clas < num_klass; ++clas)
	{
		for (int row = 0; row < m_inputImg.rows; ++row)
		{
			for (int col = 0; col < m_inputImg.cols; ++col)
			{
				int idx = m_inputImg.cols * row + col;
				Pnk.at<double>(idx, clas) = w[clas] * NNN(mu[clas], sigma[clas], m_inputImg.at<unsigned char>(row, col));
				sumPxn[idx] += Pnk.at<double>(idx, clas);
			}
		}
	}
	logSumPxn_prev = 0.0;
	for (int col = 0; col < num_klass; ++col)
	{
		for (int row = 0; row < size; ++row)
		{
			logSumPxn_prev += std::log10(sumPxn[row]);
			Pnk.at<double>(row, col) = Pnk.at<double>(row, col) / sumPxn[row];
		}
	}

	for (;;)
	{
		//update mu, sigma, w // M 
		for (int i = 0; i < w.size(); ++i)
		{
			double sum_Pnk = 0.0;
			long float znam = 0.0;
			for (int row = 0; row < m_inputImg.rows; ++row)
			{
				for (int col = 0; col < m_inputImg.cols; ++col)
				{
					int idx = m_inputImg.cols * row + col;
					znam += Pnk.at<double>(idx, i) * m_inputImg.at<unsigned char>(row, col);
					sum_Pnk += Pnk.at<double>(idx, i);
				}
			}
			w[i] = (1.0 / size) * sum_Pnk;
			mu[i] = znam / sum_Pnk;
			znam = 0.0;
			for (int row = 0; row < m_inputImg.rows; ++row)
			{
				for (int col = 0; col < m_inputImg.cols; ++col)
				{
					int idx = m_inputImg.cols * row + col;
					znam += Pnk.at<double>(idx, i) * (m_inputImg.at<unsigned char >(row, col) - mu[i])
						* (m_inputImg.at<unsigned char>(row, col) - mu[i]);
				}
			}
			sigma[i] = std::sqrt(znam / sum_Pnk);
		}

		//a E
		for (auto& i : sumPxn)
			i = 0.0;
		for (int clas = 0; clas < num_klass; ++clas)
		{
			for (int row = 0; row < m_inputImg.rows; ++row)
			{
				for (int col = 0; col < m_inputImg.cols; ++col)
				{
					int idx = m_inputImg.cols * row + col; 
					Pnk.at<double>(idx, clas) = w[clas] * NNN(mu[clas], sigma[clas], m_inputImg.at<unsigned char>(row, col));
					sumPxn[idx] += Pnk.at<double>(idx, clas);
				}
			}
		}
		double logSumPxn = 0.0;
		for (int col = 0; col < num_klass; ++col)
		{
			for (int row = 0; row < size; ++row)
			{
				logSumPxn += std::log10(sumPxn[row]); 
				Pnk.at<double>(row, col) = Pnk.at<double>(row, col) / sumPxn[row];
			}
		}

		std::cout << "w: " << w[0] << " " << w[1] << std::endl;
		std::cout << "mu: " << mu[0] << " " << mu[1] << std::endl;
		std::cout << "sigma: " << sigma[0] << " " << sigma[1] << std::endl;
		std::cout << "E: " << std::fabsf(logSumPxn - logSumPxn_prev) << std::endl << std::endl;

		if (std::fabsf(logSumPxn - logSumPxn_prev) < E) 
			break;

		logSumPxn_prev = logSumPxn;
	} //end cicle ////////////////// 

	std::vector<int> c; 
	for (auto i : mu)
		c.push_back(i);

	//choose threshold
	float z0 = std::min(mu[0] + rb*sigma[0], mu[1]);
	float z1 = std::max(mu[1] - rb*sigma[1], mu[0]);
	m_T0 = std::min(z0, z1);
	m_T1 = std::max(z0, z1);
	std::cout << "c_T0: " << (int)m_T0 << std::endl;
	std::cout << "c_T1: " << (int)m_T1 << std::endl;
	

	// Draw the histograms for B, G and R
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	cv::Mat myHist;
	m_hist.copyTo(myHist);
	for (int i = 0; i < myHist.rows; ++i)
	{
		float sum = 0.0;
		for (int j = 0; j < w.size(); ++j)
			sum += w[j] * NNN(mu[j], sigma[j], i);
		myHist.at<float>(i, 0) = sum;
	}

	cv::Mat hist;
	m_hist.copyTo(hist);
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	// Normalize the result to [ 0, histlmage.rows ]
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat()); 
	cv::normalize(myHist, myHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))), cv::Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(myHist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(myHist.at<float>(i))), cv::Scalar(255, 255, 0), 2, 8, 0);
	}
	for (auto it = c.begin(); it != c.end(); ++it)
	{
		line(histImage, 
			cv::Point(bin_w*(*it), hist_h), 
			cv::Point(bin_w*(*it), 100), cv::Scalar(0, 0, 255), 2, 8, 0);
	}
	/// Display
	cv::namedWindow("calcHist Demo EM", CV_WINDOW_AUTOSIZE);
	cv::imshow("calcHist Demo EM", histImage); 
	cv::waitKey(0);
	//////////////////////////////
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
	hist.copyTo(m_hist);

	m_cumProbHist = cv::Mat(hist.rows, hist.cols, CV_32FC1);
	m_probHist = cv::Mat(hist.rows, hist.cols, CV_32FC1);
	m_cumHist = cv::Mat(hist.rows, hist.cols, CV_32FC1);
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

	cv::Mat pop0, pop1;
	cv::Mat tmp;
	cv::Mat a(1, 1, CV_8UC1);

	unsigned char* ptr_inputImg = m_inputImg.data;
	unsigned char* ptr_threshold = m_threshold.data;
	for (int i = 0; i < m_numAllPixels; ++i)
	{
		if (m_T0 > *ptr_inputImg)
		{
			a.at<unsigned char>(0, 0) = *ptr_inputImg;
			pop0.push_back(a);
			*ptr_threshold++ = Population0;
			++mean0;
		}
		else if (m_T1 < *ptr_inputImg)
		{
			a.at<unsigned char>(0, 0) = *ptr_inputImg;
			pop1.push_back(a);
			*ptr_threshold++ = Population1;
			++mean1;
		}
		else 
			*ptr_threshold++ = UnknowPopulation;
		++ptr_inputImg;
	}

	/*int count0 = mean0;
	int count1 = mean1;
	mean0 = mean0 / m_numAllPixels;
	mean1 = mean1 / m_numAllPixels;
	m_sd0 = std::sqrtf((((1.0 - mean0) * (1.0 - mean0)) * count0) / m_numAllPixels);
	m_sd1 = std::sqrtf((((1.0 - mean1) * (1.0 - mean1)) * count1) / m_numAllPixels);*/

	m_threshold.copyTo(m_initialPopulation);
	//cv::threshold(m_inputImg, m_threshold, m_T0, 0, CV_THRESH_TOZERO);
	//cv::threshold(m_threshold, m_threshold, m_T1, 0, CV_THRESH_TRUNC);


	cv::Scalar mean, std;
	cv::meanStdDev(pop0, mean, std);
	m_sd0 = std[0];
	cv::meanStdDev(pop1, mean, std);
	m_sd1 = std[0];

	//cv::imshow("thres", m_threshold);
	//cv::waitKey(0);
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
	float sl1 = m_T1 - x;  // sr0;  m_T1 - x;
	
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

	//cv::imshow("ind0", m_indicator0);
	//cv::imshow("ind1", m_indicator1);
	//cv::waitKey(0);
	return true;
}

bool kriging::majorityFilter()
{
	if (!m_threshold.data || !m_initialPopulation.data)
		return false;

	cv::copyMakeBorder(m_initialPopulation, m_initialPopulation,
		m_radiusMF, m_radiusMF, m_radiusMF, m_radiusMF, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(UnknowPopulation));
	cv::copyMakeBorder(m_threshold, m_threshold, 
		m_radiusMF, m_radiusMF, m_radiusMF, m_radiusMF, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(UnknowPopulation));
	cv::copyMakeBorder(m_indicator0, m_indicator0,
		m_radiusMF, m_radiusMF, m_radiusMF, m_radiusMF, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0.5));
	cv::copyMakeBorder(m_indicator1, m_indicator1,
		m_radiusMF, m_radiusMF, m_radiusMF, m_radiusMF, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0.5));

	cv::Mat temp_thresh;
	m_threshold.copyTo(temp_thresh);
	float majortyThresh = (m_radiusMF * 2. + 1.) * (m_radiusMF * 2. + 1.) * m_threshMF;

	for (int row = m_radiusMF; row < temp_thresh.rows - m_radiusMF; ++row)
	{ 
		for (int col = m_radiusMF; col < temp_thresh.cols - m_radiusMF; ++col)
		{
			if (m_initialPopulation.at<uchar>(row, col) != UnknowPopulation)
			{
				int countP0 = 0;
				int countP1 = 0;
				for (int rowKernel = -m_radiusMF; rowKernel <= m_radiusMF; ++rowKernel)
				{
					for (int colKernel = -m_radiusMF; colKernel <= m_radiusMF; ++colKernel)
					{
						uchar value = temp_thresh.at<uchar>(row + rowKernel, col + colKernel);
						if (value == Population0)
							++countP0;
						else if (value == Population1)
							++countP1;
					}
				}
				uchar value = m_threshold.at<uchar>(row, col);
				if (countP0 > majortyThresh && value != Population0)
				{
					m_threshold.at<uchar>(row, col) = Population0;
					m_indicator0.at<float>(row, col) = 1.0;
					m_indicator1.at<float>(row, col) = 1.0;
				}
				else if (countP1 > majortyThresh && value != Population1)
				{
					m_threshold.at<uchar>(row, col) = Population0;
					m_indicator0.at<float>(row, col) = 1.0;
					m_indicator1.at<float>(row, col) = 1.0;
				}
			}
		}
	}

	cv::Rect rect(m_radiusMF, m_radiusMF, m_inputImg.cols, m_inputImg.rows);
	m_initialPopulation  = m_initialPopulation(rect);
	m_threshold = m_threshold(rect);
	m_indicator0 = m_indicator0(rect);
	m_indicator1 = m_indicator1(rect); 

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

	std::cout << "c_T0: " << (int)m_T0 << std::endl;
	std::cout << "c_T1: " << (int)m_T1 << std::endl;
}



fixedWindowKriging::fixedWindowKriging(int radiusKriging, int radiusMF, float threshMF) 
: kriging(radiusMF, threshMF), m_radiusKriging(radiusKriging) {}


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
	for (int row = -m_radiusKriging; row <= m_radiusKriging; ++row)
	{
		for (int col = -m_radiusKriging; col <= m_radiusKriging; ++col)							//			   00
		{																						//		 01 02 03 04 05																	//	  11 12 13 xx .. .. ..
			if (std::sqrt(row*row + col *col) <= m_radiusKriging && !(row == 0 && col == 0))	//		 06 07 08 09 10
				m_krigingKernelIndex.push_back(std::pair<int, int>(row, col));					//	  11 12 13 28 14 15 16
		}																						//		 17 18 19 20 21
	}																							//		 22 23 24 25 26																									//			   27
	m_krigingKernelIndex.push_back(std::pair<int, int>(0, 0));									//			   27
	m_numElemUnderWindow = m_krigingKernelIndex.size();


	//create accelerate
	for (auto it_first : m_krigingKernelIndex)
	{
		for (auto it_second : m_krigingKernelIndex)
		{
			// calc radius
			// if radius not in set
			// then add to set with inverse coordinate
		}

	}
}

float fixedWindowKriging::covariance(const cv::Mat seq0, const cv::Mat seq1) const
{
	float covar = ((cv::sum(seq0.mul(seq1)) / m_numAllPixels) - (cv::mean(seq0) * cv::mean(seq1)))[0];
	return ((cv::sum(seq0.mul(seq1)) / m_numAllPixels) - (cv::mean(seq0) * cv::mean(seq1)))[0];
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
		for (int i = 0; i < system_size; ++i) // /2
		{
			krigingSystemRight.at<float>(i, 0) = covariance(sequencesMatrix.col(system_size), sequencesMatrix.col(i));
			//krigingSystemRight.at<float>(system_size - 1 - i, 0) = krigingSystemRight.at<float>(i, 0);
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
				float val = covariance(sequencesMatrix.col(row), sequencesMatrix.col(col));
				krigingSystemLeft.at<float>(row, col) = val;
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
	int kernelSize = 2 * m_radiusKriging + 1;
	cv::Mat krigingKernel(kernelSize, kernelSize, CV_32FC1, cv::Scalar(0.0));
	int size_system = m_numElemUnderWindow - 1;
	for (int i = 0; i < size_system; ++i)
		krigingKernel.at<float>(m_krigingKernelIndex[i].first + m_radiusKriging, m_krigingKernelIndex[i].second + m_radiusKriging) = weightsMatrix.at<float>(i, 0);

	return krigingKernel;
}

void fixedWindowKriging::setKrigingKernel(float sigma)
{
	int kernelSize = 2 * m_radiusKriging + 1;
	m_krigingKernel0.create(kernelSize, kernelSize, CV_32FC1);
	m_krigingKernel1.create(kernelSize, kernelSize, CV_32FC1);

	for (int x = -m_radiusKriging; x < -m_radiusKriging + kernelSize; x++)
	{
		for (int y = -m_radiusKriging; y < -m_radiusKriging + kernelSize; y++)
		{
			m_krigingKernel0.at<float>(x + m_radiusKriging, y + m_radiusKriging) = (1.0 / (sigma * std::sqrt(2 *3.1481))) *  exp(-((x * x + y * y) / (2 * sigma * sigma)));;
			m_krigingKernel1.at<float>(x + m_radiusKriging, y + m_radiusKriging) = m_krigingKernel0.at<float>(x + m_radiusKriging, y + m_radiusKriging);
		}
	}
	m_krigingKernel0.at<float>(m_radiusKriging, m_radiusKriging) = 0.0;
	m_krigingKernel1.at<float>(m_radiusKriging, m_radiusKriging) = 0.0;
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
	int kernelSize = 2 * m_radiusKriging + 1;
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

	int board = m_radiusKriging;

	cv::copyMakeBorder(m_indicator0, indicator0, board, board, board, board, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0.5));
	cv::copyMakeBorder(m_indicator1, indicator1, board, board, board, board, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0.5));

	if ((indicator0.rows != indicator1.rows) || (indicator0.cols != indicator1.cols))
		return false;

	//int num_pixels4sequnec = (indicator0.rows - board) * (indicator0.cols - board);
	int num_pixels4sequnec = m_numAllPixels;
	cv::Mat sequences0(num_pixels4sequnec, m_numElemUnderWindow, CV_32FC1);
	cv::Mat sequences1(num_pixels4sequnec, m_numElemUnderWindow, CV_32FC1);
	int counter = 0;
	for (int row = m_radiusKriging; row < indicator0.rows - m_radiusKriging; ++row)
	{
		for (int col = m_radiusKriging; col < indicator0.cols - m_radiusKriging; ++col)
		{
			int i = 0;
			for (const auto coord : m_krigingKernelIndex)
			{
				sequences0.at<float>(counter, i) = indicator0.at<float>(row + coord.first, col + coord.second);
				sequences1.at<float>(counter, i) = indicator1.at<float>(row + coord.first, col + coord.second);
				//std::cout << row << " " << col << std::endl;
				i++;
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

	cv::filter2D(m_indicator0, m_probabilityPopulation0, CV_32FC1, m_krigingKernel0); // , cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
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

	//cv::imwrite("..//images//output//tests//beforeMF_2.png", m_threshold); //
	return true;
}



double getMSSIM(const cv::Mat& i1, const cv::Mat& i2, const cv::Size& win_size)
{
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_64FC1;

	cv::Mat I1, I2;
	i1.convertTo(I1, d);           // cannot calculate on one byte large values
	i2.convertTo(I2, d);

	cv::Mat I2_2 = I2.mul(I2);        // I2^2
	cv::Mat I1_2 = I1.mul(I1);        // I1^2
	cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

									   /*************************** END INITS **********************************/

	cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
	cv::blur(I1, mu1, win_size);
	cv::blur(I1, mu2, win_size);
	//cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	//cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu1_mu2 = mu1.mul(mu2);

	cv::Mat sigma1_2, sigma2_2, sigma12;

	//cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	cv::blur(I1_2, sigma1_2, win_size);
	sigma1_2 -= mu1_2;

	//cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	cv::blur(I2_2, sigma2_2, win_size);
	sigma2_2 -= mu2_2;

	//cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
	cv::blur(I1_I2, sigma12, win_size);
	sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	cv::Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	cv::Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	cv::Scalar mssim = cv::mean(ssim_map); // mssim = average of ssim map
	double res = mssim[0];
	return res;
}


double calcGradientsSumMetric(const cv::Mat& img, const cv::Mat& segment)
{ // in 8UC1
	cv::Mat imgf;
	img.convertTo(imgf, CV_64FC1, 1. / 255.);

	cv::Mat Gradient;
	cv::Mat sbx, sby;
	cv::Sobel(imgf, sbx, CV_64FC1, 0, 1);
	cv::Sobel(imgf, sby, CV_64FC1, 1, 0);
	cv::magnitude(sbx, sby, Gradient);

	cv::Mat MrfGr;
	cv::Mat kernel(3, 3, CV_8UC1, 1);
	cv::morphologyEx(segment, MrfGr, cv::MORPH_GRADIENT, kernel);
	MrfGr.convertTo(MrfGr, CV_64FC1, 1./255.);

	cv::Mat Derive = MrfGr.mul(Gradient);
	
	cv::Mat Dat;
	double min, max;
	cv::minMaxLoc(Derive, &min, &max);
	double thresh = 0.25 * (max - min);
	cv::threshold(Derive, Dat, thresh, 0, cv::THRESH_TOZERO);
	cv::threshold(Derive, Derive, thresh, 0, cv::THRESH_TOZERO);

	/*cv::imshow("Gradient", Gradient);
	cv::imshow("MrfGradient", MrfGr);
	cv::imshow("Derive", Derive);
	cv::imshow("DeriveAfterThresh", Dat);
	cv::waitKey(0);*/

	cv::Mat out_metric;
	cv::pow(Derive, 2.0, out_metric);
	cv::Scalar abs_value = cv::sum(out_metric);
	double tmp = abs_value[0] / img.total();

	return tmp;

}

double correlation(const cv::Mat &image_1, const cv::Mat &image_2)  
{ // in 8UC1
	const double L = 256.0;
	const double k2 = 0.03;
	const double c2 = std::pow(k2 * L, 2.0);
	const double c3 = c2 / 2.0;

	// convert data-type to "float"
	cv::Mat im_float_1;
	image_1.convertTo(im_float_1, CV_64F);
	cv::Mat im_float_2;
	image_2.convertTo(im_float_2, CV_64F);

	int n_pixels = im_float_1.rows * im_float_1.cols;

	// Compute mean and standard deviation of both images
	cv::Scalar im1_Mean, im1_Std, im2_Mean, im2_Std;
	cv::meanStdDev(im_float_1, im1_Mean, im1_Std);
	cv::meanStdDev(im_float_2, im2_Mean, im2_Std);

	// Compute covariance and correlation coefficient
	double covar =  (im_float_1 - im1_Mean).dot(im_float_2 - im2_Mean) / n_pixels;
	double correl = (covar + c3) / ((im1_Std[0] * im2_Std[0]) + c3);

	return correl;
}

double ICV(const cv::Mat& img, const cv::Mat& segment)
{
	cv::Scalar img_stdDevs, img_means;
	cv::meanStdDev(img, img_means, img_stdDevs);

	cv::Scalar seg_stdDevs, seg_means;
	cv::meanStdDev(img, seg_means, seg_stdDevs, segment);
	int PoreCount = cv::countNonZero(segment);
	int Total = img.total();

	double VarPore = seg_stdDevs[0] * seg_stdDevs[0];
	double VarTotal = img_stdDevs[0] * img_stdDevs[0];

	double metric = ((double)PoreCount / (double)Total) * (VarPore / VarTotal);
	return metric;
}

double otsu_parametr(const cv::Mat& img, const cv::Mat& segment)
{
	cv::Scalar stdDevs_1, means_1;
	cv::meanStdDev(img, means_1, stdDevs_1, segment);

	int Total = img.total();
	int Count_1 = cv::countNonZero(segment);
	int Count_2 = Total - Count_1;
	double W1 = (double)Count_1 / (double)Total;
	double W2 = (double)Count_2 / (double)Total;

	cv::Mat inverse_segment;
	cv::bitwise_not(segment, inverse_segment);
	cv::Scalar stdDevs_2, means_2;
	cv::meanStdDev(img, means_2, stdDevs_2, inverse_segment);

	double Var_1 = stdDevs_1[0] * stdDevs_1[0];
	double Var_2 = stdDevs_2[0] * stdDevs_2[0];

	double metric = (W1 * Var_1) + (W2 * Var_2);
	return metric;
}

double MSSIM(const cv::Mat& img, const cv::Mat& segment, const cv::Size& win_size)
{
	cv::Mat inverse_segment;
	cv::bitwise_not(segment, inverse_segment);

	cv::Mat rock = img.mul(segment / 255.);
	cv::Mat background = img.mul(inverse_segment / 255.);
	//cv::imshow("rock", rock);
	//cv::imshow("back", background);
	//cv::waitKey(0);

	double mssim_rock = getMSSIM(rock, segment, win_size);
	double mssim_background = getMSSIM(background, inverse_segment, win_size);

	double mssim = mssim_rock * mssim_background;
	return mssim;
}

double GVC(const cv::Mat& img, const cv::Mat& segment)
{
	cv::Mat inverse_segement;
	cv::bitwise_not(segment, inverse_segement);

	cv::Scalar means_1;
	means_1 = cv::mean(img, segment);
	cv::Scalar means_2;
	means_2 = cv::mean(img, inverse_segement);

	double diff_mean = std::abs(means_1[0] - means_2[0]);
	double sum_mean = means_1[0] + means_2[0];
	double gvc = diff_mean / sum_mean;

	return gvc;
}
