// This file has been generated by Py++.

// BornAgain: simulate and fit scattering at grazing incidence 
//! @brief automatically generated boost::python code for PythonCoreAPI  

#include "Macros.h"
GCC_DIAG_OFF(unused-parameter);
GCC_DIAG_OFF(missing-field-initializers);
#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp"
GCC_DIAG_ON(unused-parameter);
GCC_DIAG_ON(missing-field-initializers);
#include "PythonFitList.h"
#include "vector_string_t.pypp.h"

namespace bp = boost::python;

void register_vector_string_t_class(){

    { //::std::vector< std::string >
        typedef bp::class_< std::vector< std::string > > vector_string_t_exposer_t;
        vector_string_t_exposer_t vector_string_t_exposer = vector_string_t_exposer_t( "vector_string_t" );
        bp::scope vector_string_t_scope( vector_string_t_exposer );
        vector_string_t_exposer.def( bp::vector_indexing_suite< ::std::vector< std::string >, true >() );
    }

}