#include <iostream>
#include "opencv2\opencv.hpp"
#include "kriging.h"
#include < limits.h >  //Для PATH_MAX
#include < stdio.h >   //Для printf
#include <direct.h>

int main()
{
	fixedWindowKriging k;
	k.read("..//images//input//test.png");
	k.calcHist();
	k.setT(87, 127);
	k.thresholding();
	k.calcIndicator();
	k.majorityFilter();
	k.calcCovarianceMatrix();


	return 0;
}