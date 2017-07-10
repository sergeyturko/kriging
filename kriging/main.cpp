#include "kriging.h"


int main()
{
	fixedWindowKriging test1;

	test1.read("..//images//input//FB_2_1_1__rec3002.png");

	std::vector<double> t = { 60, 190};
	test1.Kmeans(t);

	test1.calcHist();
	test1.Otsu(2);
	test1.calcHist();

	//for (int i = 0; i <= 255; ++i)
	//{
	//	float val = test1.calcKapurEtrophy(i);
	//	std::cout << val << std::endl;
	//}
	//test1.chooseThreshold(0.03);
	//std::system("pause");

	//return 0;

	//test1.setT(80, 95);
	test1.thresholding();
	test1.calcIndicator();
	test1.majorityFilter();
	test1.calcCovarianceMatrix();
	test1.calcProbability();
	test1.majorityFilter();
	test1.write("321", false);
	test1.show();

	return 0;
}