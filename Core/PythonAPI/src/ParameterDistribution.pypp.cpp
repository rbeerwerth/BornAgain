// This file has been generated by Py++.

// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Automatically generated boost::python code for BornAgain Python bindings
//! @brief     Automatically generated boost::python code for BornAgain Python bindings
//!
//! @homepage  http://bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Juelich GmbH 2015
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   C. Durniak, M. Ganeva, G. Pospelov, W. Van Herck, J. Wuttke
//
// ************************************************************************** //

#include "Macros.h"
GCC_DIAG_OFF(unused-parameter)
GCC_DIAG_OFF(missing-field-initializers)
#include "boost/python.hpp"
GCC_DIAG_ON(unused-parameter)
GCC_DIAG_ON(missing-field-initializers)
#include "PythonCoreList.h"
#include "ParameterDistribution.pypp.h"

namespace bp = boost::python;

void register_ParameterDistribution_class(){

    { //::ParameterDistribution
        typedef bp::class_< ParameterDistribution, bp::bases< IParameterized > > ParameterDistribution_exposer_t;
        ParameterDistribution_exposer_t ParameterDistribution_exposer = ParameterDistribution_exposer_t( "ParameterDistribution", bp::init< std::string const &, IDistribution1D const &, std::size_t, bp::optional< double, AttLimits const & > >(( bp::arg("par_name"), bp::arg("distribution"), bp::arg("nbr_samples"), bp::arg("sigma_factor")=0.0, bp::arg("limits")=::AttLimits( ) )) );
        bp::scope ParameterDistribution_scope( ParameterDistribution_exposer );
        ParameterDistribution_exposer.def( bp::init< std::string const &, IDistribution1D const &, std::size_t, double, double >(( bp::arg("par_name"), bp::arg("distribution"), bp::arg("nbr_samples"), bp::arg("xmin"), bp::arg("xmax") )) );
        ParameterDistribution_exposer.def( bp::init< ParameterDistribution const & >(( bp::arg("other") )) );
        { //::ParameterDistribution::getDistribution
        
            typedef ::IDistribution1D const * ( ::ParameterDistribution::*getDistribution_function_type)(  ) const;
            
            ParameterDistribution_exposer.def( 
                "getDistribution"
                , getDistribution_function_type( &::ParameterDistribution::getDistribution )
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ParameterDistribution::getLimits
        
            typedef ::AttLimits ( ::ParameterDistribution::*getLimits_function_type)(  ) const;
            
            ParameterDistribution_exposer.def( 
                "getLimits"
                , getLimits_function_type( &::ParameterDistribution::getLimits ) );
        
        }
        { //::ParameterDistribution::getMainParameterName
        
            typedef ::std::string ( ::ParameterDistribution::*getMainParameterName_function_type)(  ) const;
            
            ParameterDistribution_exposer.def( 
                "getMainParameterName"
                , getMainParameterName_function_type( &::ParameterDistribution::getMainParameterName )
                , "get the main parameter's name." );
        
        }
        { //::ParameterDistribution::getMaxValue
        
            typedef double ( ::ParameterDistribution::*getMaxValue_function_type)(  ) const;
            
            ParameterDistribution_exposer.def( 
                "getMaxValue"
                , getMaxValue_function_type( &::ParameterDistribution::getMaxValue ) );
        
        }
        { //::ParameterDistribution::getMinValue
        
            typedef double ( ::ParameterDistribution::*getMinValue_function_type)(  ) const;
            
            ParameterDistribution_exposer.def( 
                "getMinValue"
                , getMinValue_function_type( &::ParameterDistribution::getMinValue ) );
        
        }
        { //::ParameterDistribution::getNbrSamples
        
            typedef ::std::size_t ( ::ParameterDistribution::*getNbrSamples_function_type)(  ) const;
            
            ParameterDistribution_exposer.def( 
                "getNbrSamples"
                , getNbrSamples_function_type( &::ParameterDistribution::getNbrSamples )
                , "get number of samples for this distribution." );
        
        }
        { //::ParameterDistribution::getSigmaFactor
        
            typedef double ( ::ParameterDistribution::*getSigmaFactor_function_type)(  ) const;
            
            ParameterDistribution_exposer.def( 
                "getSigmaFactor"
                , getSigmaFactor_function_type( &::ParameterDistribution::getSigmaFactor )
                , "get the sigma factor." );
        
        }
        { //::ParameterDistribution::linkParameter
        
            typedef ::ParameterDistribution & ( ::ParameterDistribution::*linkParameter_function_type)( ::std::string ) ;
            
            ParameterDistribution_exposer.def( 
                "linkParameter"
                , linkParameter_function_type( &::ParameterDistribution::linkParameter )
                , ( bp::arg("par_name") )
                , bp::return_value_policy< bp::reference_existing_object >()
                , "Overload assignment operator." );
        
        }
        { //::ParameterDistribution::operator=
        
            typedef ::ParameterDistribution & ( ::ParameterDistribution::*assign_function_type)( ::ParameterDistribution const & ) ;
            
            ParameterDistribution_exposer.def( 
                "assign"
                , assign_function_type( &::ParameterDistribution::operator= )
                , ( bp::arg("other") )
                , bp::return_self< >()
                , "Overload assignment operator." );
        
        }
    }

}
