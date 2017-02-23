#include <iostream>
#include "opencv2\opencv.hpp"
#include "kriging.h"

int main()
{
	indicator_kriging k;
	k.read("");
	k.setT(25, 68);
	k.calcHist();
	k.treshoold();
	k.calcCovariance();
	k.calcIndicator();
	//cv::namedWindow("test", 0);
	cv::waitKey(0);
	return 0;
}