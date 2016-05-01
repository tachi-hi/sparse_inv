#include<Python.h>
#include<numpy/arrayobject.h>

#include<boost/numeric/ublas/vector.hpp>
#include<pyublas/numpy.hpp>

#include"proc.h"

using namespace std;
using namespace boost::python;
using namespace boost::numeric;
using namespace pyublas;

// ----------------------------------------------------------------------------
BOOST_PYTHON_MODULE( myTest ){
	numeric::array::set_module_and_type( "numpy", "ndarray" );

	boost::python::def( "logp", &LOGP );
	import_array();
}
