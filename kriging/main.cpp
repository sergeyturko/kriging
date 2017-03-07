#include <iostream>
#include "opencv2\opencv.hpp"
#include "kriging.h"


int main()
{
	fixedWindowKriging test1;

	test1.read("..//images//input//FB_2_1_1__rec3002.png");
	test1.calcHist();
	test1.setT(94, 126);
	test1.thresholding();
	test1.calcIndicator();
	test1.majorityFilter();
	test1.calcCovarianceMatrix();
	test1.calcProbability();
	test1.majorityFilter();
	test1.write("test1");
	test1.show();

	return 0;
}