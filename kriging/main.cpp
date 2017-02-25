#include <iostream>
#include "opencv2\opencv.hpp"
#include "kriging.h"
#include < limits.h >  //Для PATH_MAX
#include < stdio.h >   //Для printf
#include <direct.h>

int main()
{
	kriging k;
	k.read("..//images//input//FB_2_1_1__rec3002.png");
	k.calcHist();
	k.setT(117, 119);
	k.thresholding();
	k.calcIndicator();
	k.majorityFilter();


	return 0;
}