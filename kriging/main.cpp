#include "kriging.h"


int main()
{
	fixedWindowKriging test1;
	fixedWindowKriging test2;

	test1.read("..//images//input//FB_2_1_1__rec3002.png");
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