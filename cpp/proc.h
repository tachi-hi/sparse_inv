#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include<boost/numeric/ublas/vector.hpp>
#include<Python.h>
#include<numpy/arrayobject.h>

#include<boost/numeric/ublas/vector.hpp>
#include<pyublas/numpy.hpp>

using namespace std;
using namespace boost::numeric;

boost::numeric::ublas::vector<double>
LOGP(
	const ublas::matrix<double> &A,
	const ublas::matrix<double> &Xi_tmp,
	const boost::numeric::ublas::vector<double> &y,
	double weight,
	const int iter);
