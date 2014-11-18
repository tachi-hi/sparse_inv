#include<Python.h>
#include<numpy/arrayobject.h>

#include<boost/python.hpp>
#include<boost/python/numeric.hpp>
#include<boost/python/extract.hpp>

#include<boost/numeric/ublas/vector.hpp>
#include<pyublas/numpy.hpp>

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>


using namespace std;
using namespace boost::python;
using namespace boost::numeric;
using namespace pyublas;

double logPenalty(const double x){
	return log(x * x + 1e-300);
}

// ----------------------------------------------------------------------------
boost::numeric::ublas::vector<double>
LOGP(
	const ublas::matrix<double> &A,
	const ublas::matrix<double> &Xi_tmp,
	const boost::numeric::ublas::vector<double> &y,
	double weight,
	const int iter)
{
	ublas::matrix<double> Xi = Xi_tmp;
	boost::numeric::ublas::vector<double> yA = prod(trans(A), y );

	std::vector<unsigned int> foreach_array;
	boost::numeric::ublas::vector<double> x( A.size2() ); // ret value
	for(unsigned int i = 0; i != x.size(); ++i){
		foreach_array.push_back( i );
		x(i) = 0;
	}

	// -------------------------------------------------------
	const auto compute_m = [&](
		const unsigned int j,
		const ublas::matrix<double> &A,
		const ublas::matrix<double> &Xi,
		const boost::numeric::ublas::vector<double> &y,
		const boost::numeric::ublas::vector<double> &x
		) -> double
	{
		double Xix = 0; 
		for(unsigned int i = 0; i != Xi.size2(); ++i)
			if(i != j) 
				Xix += Xi(i,j) * x(i);

		return (yA(j) - Xix) / Xi(j,j);
	};

	// -------------------------------------------------------
	const auto compute_Delta = [&](
		const double m_k, 
		const unsigned int k
		) -> double
	{
		double Xix = 0; 
		for(unsigned int i = 0; i != Xi.size2(); ++i)
			if(i != k) 
				Xix += Xi(i,k) * x(i);

		const double logp = logPenalty( m_k );

		return ((2 * yA(k) * m_k) + (Xi(k, k) * m_k * m_k) - (2 * m_k * Xix))/2 - weight * logp;
	};

	// -------------------------------------------------------
	for(int it = 0; it != iter; ++it){

		random_shuffle( foreach_array.begin(), foreach_array.end() ); 

		for(auto ind = foreach_array.begin(); ind != foreach_array.end(); ++ind)
		{
			unsigned int j = *ind;
			const double m = compute_m(j, A, Xi, y, x);

			const double D = m * m - 4 * weight / Xi(j,j);
			if(D < 0){
				x(j) = 0;
			}else{
				const double tmp = m - weight / Xi(j,j)/ m;
				const double delta = compute_Delta(tmp, j);
				if(delta > 0){
					x(j) = tmp;
				}else{
					x(j) = 0;
				}
			}
		}
	}
	return x;
}

// ----------------------------------------------------------------------------
BOOST_PYTHON_MODULE( myTest ){
	numeric::array::set_module_and_type( "numpy", "ndarray" );

	boost::python::def( "logp", &LOGP );
	import_array();
}


