#include <iostream>
#include "opencv2\opencv.hpp"
#include "kriging.h"
#include < limits.h >  //Äëÿ PATH_MAX
#include < stdio.h >   //Äëÿ printf
#include <direct.h>

int main()
{
	fixedWindowKriging k;
	k.read("..//images//input//FB_2_1_1__rec3002.png");
	k.calcHist();
	k.setT(94, 126);
	k.thresholding();
	k.calcIndicator();
	k.majorityFilter();
	k.calcCovarianceMatrix();
	k.calcProbability();
	k.majorityFilter();
	k.write("test1");

	return 0;
}