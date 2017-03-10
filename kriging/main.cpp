#include <iostream>
#include "opencv2\opencv.hpp"
#include "kriging.h"


int main()
{
	fixedWindowKriging test1;

	test1.read("..//images//tests//test_25.0_GN.png");
	test1.calcHist();
	test1.setT(150, 166);
	test1.thresholding();
	test1.calcIndicator();
	test1.majorityFilter();
	test1.calcCovarianceMatrix();
	test1.calcProbability();
	test1.majorityFilter();
	test1.write("testGN25", true);
	test1.show();

	return 0;
}