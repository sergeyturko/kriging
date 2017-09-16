#include "kriging.h"

#include <iostream>
#include <fstream>

double ssim[100][100];
double grad[100][100];

void runTest()
{
	std::ofstream file;
	file.open("D:\\ProgWork\\kriging\\testTHRESH.txt"); //ATTENTION

	for (int T0 = 50; T0 < 130; ++T0)
	{
		for (int T1 = T0 + 2; T1 < 130; ++T1)
		{
			fixedWindowKriging test1;
			test1.read("..//images//input//bhi_2_2.32um_voi_0751.png");
			test1.calcHist();
			test1.setT(T0, T1);
			test1.thresholding();
			test1.calcIndicator();
			test1.majorityFilter();
			test1.calcCovarianceMatrix();
			test1.calcProbability();
			test1.majorityFilter();

			double gradient_metric = calcGradientsSumMetric(test1.m_inputImg, test1.m_threshold);
			double ssim_metric = correlation(test1.m_inputImg, test1.m_threshold);

			file << "T0: " << (int)test1.m_T0 << " T1: " << (int)test1.m_T1 << std::endl;
			file << "grad: " << gradient_metric << std::endl;
			file << "ssim: " << ssim_metric << std::endl;
			file << std::endl;

			std::cout << "T0: " << (int)test1.m_T0 << " T1: " << (int)test1.m_T1 << std::endl;
			std::cout << "grad: " << gradient_metric << std::endl;
			std::cout << "ssim: " << ssim_metric << std::endl;
			std::cout << std::endl;

		}
	}
	file.close();
}


int main()
{
	//runTest();
	//return 0;

	fixedWindowKriging test1;
	fixedWindowKriging test2;

	//test1.read("..//images//input//AC2_1_HQ__rec2501.png");
	//test1.read("..//images//input//FB_2_1_1__rec3002.png");
	test1.read("..//images//input//bhi_2_2.32um_voi_0751.png");
	test1.calcHist();
	//test1.chooseThreshold(0.03);
	//test1.EM(2, 1.5);
	std::vector<double> t = { 61, 100 };
	//test1.Kmeans(t);
	//test1.setT(87, 139);
	//test1.setT(84, 128);
	test1.setT(61, 95);
	//test1.setT(75, 85);
	test1.thresholding();
	test1.calcIndicator();
	test1.majorityFilter();
	test1.calcCovarianceMatrix();
	test1.calcProbability();
	test1.majorityFilter();
	//test1.write("test_with_covariance_2radiusBorader", true);

	cv::Mat otsu;
	cv::threshold(test1.m_inputImg, otsu, 0, 255, CV_THRESH_BINARY | cv::THRESH_OTSU);
	cv::imshow("in", test1.m_inputImg);
	cv::imshow("otsu", otsu);

	cv::Mat KMeans;
	cv::threshold(test1.m_inputImg, KMeans, 80, 255, CV_THRESH_BINARY);
	cv::imshow("KM", KMeans);
//	cv::waitKey(0);
	double gradient_metric_otsu = calcGradientsSumMetric(test1.m_inputImg, otsu);
	double corr_metric_otsu = correlation(test1.m_inputImg, otsu);
	double icv_metric_otsu = ICV(test1.m_inputImg, otsu);
	double otsuparam_metric_otsu = otsu_parametr(test1.m_inputImg, otsu);
	double mssimbase_metric_otsu = MSSIM(test1.m_inputImg, otsu);
	double gvc_metric_otsu = GVC(test1.m_inputImg, otsu);

	double gradient_metric_KM = calcGradientsSumMetric(test1.m_inputImg, KMeans);
	double corr_metric_KM = correlation(test1.m_inputImg, KMeans);
	double icv_metric_KM = ICV(test1.m_inputImg, KMeans);
	double otsuparam_metric_KM = otsu_parametr(test1.m_inputImg, KMeans);
	double mssimbase_metric_KM = MSSIM(test1.m_inputImg, KMeans);
	double gvc_metric_KM = GVC(test1.m_inputImg, KMeans);

	test1.show();
	double gradient_metric = calcGradientsSumMetric(test1.m_inputImg, test1.m_threshold);
	double corr_metric = correlation(test1.m_inputImg, test1.m_threshold);
	double icv_metric = ICV(test1.m_inputImg, test1.m_threshold);
	double otsuparam_metric = otsu_parametr(test1.m_inputImg, test1.m_threshold);
	double mssimbase_metric = MSSIM(test1.m_inputImg, test1.m_threshold);
	double gvc_metric = GVC(test1.m_inputImg, test1.m_threshold);

	std::cout << std::endl;
	std::cout << "otsu grad: " << gradient_metric_otsu << std::endl;
	std::cout << "otsu corr: " << corr_metric_otsu << std::endl;
	std::cout << "otsu icv: " << icv_metric_otsu << std::endl;
	std::cout << "otsu otsuparam: " << otsuparam_metric_otsu << std::endl;
	std::cout << "otsu mssimbase: " << mssimbase_metric_otsu << std::endl;
	std::cout << "otsu gvc: " << gvc_metric_otsu << std::endl << std::endl;

	std::cout << "KM grad: " << gradient_metric_KM << std::endl;
	std::cout << "KM corr: " << corr_metric_KM << std::endl;
	std::cout << "KM icv: " << icv_metric_KM << std::endl;
	std::cout << "KM otsuparam: " << otsuparam_metric_KM << std::endl;
	std::cout << "KM mssimbase: " << mssimbase_metric_KM << std::endl;
	std::cout << "KM gvc: " << gvc_metric_KM << std::endl << std::endl;
	
	std::cout << "kr grad: " << gradient_metric << std::endl;
	std::cout << "kr corr: " << corr_metric << std::endl;
	std::cout << "kr icv: " << icv_metric << std::endl;
	std::cout << "kr otsuparam: " << otsuparam_metric << std::endl;
	std::cout << "kr mssimbase: " << mssimbase_metric << std::endl;
	std::cout << "kr gvc: " << gvc_metric << std::endl << std::endl;
	std::system("pause");

	//test1.show();


	//std::vector<double> t = { 60, 190 };
	//test1.Kmeans(t);
	//test1.EM(2);

	return 0;




	test2.read("..//images//input//FB_2_1_1__rec3002.png");

	/*std::vector<double> t = { 60, 190};
	test1.Kmeans(t);

	test1.calcHist();
	test1.Otsu(2);*/

	test1.calcHist();
	test1.chooseThreshold(0.1);
	test2.calcHist();
	test2.chooseThreshold(0.1);
	//test1.setT(80, 95);

	test1.thresholding();
	test1.calcIndicator();
	test1.majorityFilter();
//	test2.thresholding();
///	test2.calcIndicator();
//	test2.majorityFilter();
//	test2.calcCovarianceMatrix();
	test1.setKrigingKernel(10.5);
	test1.calcProbability();
	test1.majorityFilter();
//	test2.calcProbability();
//	test2.majorityFilter();
//	test1.write("t01s105", true);
	test1.show();
//	test2.show();


	return 0;
}