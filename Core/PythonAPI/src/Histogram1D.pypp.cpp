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
#include "Histogram1D.pypp.h"

namespace bp = boost::python;

struct Histogram1D_wrapper : Histogram1D, bp::wrapper< Histogram1D > {

    Histogram1D_wrapper(int nbinsx, double xlow, double xup )
    : Histogram1D( nbinsx, xlow, xup )
      , bp::wrapper< Histogram1D >(){
        // constructor
    
    }

    Histogram1D_wrapper(int nbinsx, ::std::vector< double > const & xbins )
    : Histogram1D( nbinsx, boost::ref(xbins) )
      , bp::wrapper< Histogram1D >(){
        // constructor
    
    }

    Histogram1D_wrapper(::IAxis const & axis )
    : Histogram1D( boost::ref(axis) )
      , bp::wrapper< Histogram1D >(){
        // constructor
    
    }

    Histogram1D_wrapper(::OutputData< double > const & data )
    : Histogram1D( boost::ref(data) )
      , bp::wrapper< Histogram1D >(){
        // constructor
    
    }

    virtual ::std::size_t getRank(  ) const  {
        if( bp::override func_getRank = this->get_override( "getRank" ) )
            return func_getRank(  );
        else{
            return this->Histogram1D::getRank(  );
        }
    }
    
    ::std::size_t default_getRank(  ) const  {
        return Histogram1D::getRank( );
    }

    virtual ::std::size_t getTotalNumberOfBins(  ) const  {
        if( bp::override func_getTotalNumberOfBins = this->get_override( "getTotalNumberOfBins" ) )
            return func_getTotalNumberOfBins(  );
        else{
            return this->IHistogram::getTotalNumberOfBins(  );
        }
    }
    
    ::std::size_t default_getTotalNumberOfBins(  ) const  {
        return IHistogram::getTotalNumberOfBins( );
    }

    virtual ::IAxis const * getXaxis(  ) const  {
        if( bp::override func_getXaxis = this->get_override( "getXaxis" ) )
            return func_getXaxis(  );
        else{
            return this->IHistogram::getXaxis(  );
        }
    }
    
    ::IAxis const * default_getXaxis(  ) const  {
        return IHistogram::getXaxis( );
    }

    virtual double getXaxisValue( ::std::size_t binGlobalIndex ) {
        if( bp::override func_getXaxisValue = this->get_override( "getXaxisValue" ) )
            return func_getXaxisValue( binGlobalIndex );
        else{
            return this->IHistogram::getXaxisValue( binGlobalIndex );
        }
    }
    
    double default_getXaxisValue( ::std::size_t binGlobalIndex ) {
        return IHistogram::getXaxisValue( binGlobalIndex );
    }

    virtual ::IAxis const * getYaxis(  ) const  {
        if( bp::override func_getYaxis = this->get_override( "getYaxis" ) )
            return func_getYaxis(  );
        else{
            return this->IHistogram::getYaxis(  );
        }
    }
    
    ::IAxis const * default_getYaxis(  ) const  {
        return IHistogram::getYaxis( );
    }

    virtual double getYaxisValue( ::std::size_t binGlobalIndex ) {
        if( bp::override func_getYaxisValue = this->get_override( "getYaxisValue" ) )
            return func_getYaxisValue( binGlobalIndex );
        else{
            return this->IHistogram::getYaxisValue( binGlobalIndex );
        }
    }
    
    double default_getYaxisValue( ::std::size_t binGlobalIndex ) {
        return IHistogram::getYaxisValue( binGlobalIndex );
    }

    virtual void reset(  ) {
        if( bp::override func_reset = this->get_override( "reset" ) )
            func_reset(  );
        else{
            this->IHistogram::reset(  );
        }
    }
    
    void default_reset(  ) {
        IHistogram::reset( );
    }

};

void register_Histogram1D_class(){

    { //::Histogram1D
        typedef bp::class_< Histogram1D_wrapper, bp::bases< IHistogram >, boost::noncopyable > Histogram1D_exposer_t;
        Histogram1D_exposer_t Histogram1D_exposer = Histogram1D_exposer_t( "Histogram1D", bp::init< int, double, double >(( bp::arg("nbinsx"), bp::arg("xlow"), bp::arg("xup") ), "Constructor for fix bin size histograms. @param nbinsx number of bins @param xlow low edge of the first bin @param xup upper edge of the last bin \n\n:Parameters:\n  - 'nbinsx' - number of bins\n  - 'xlow' - low edge of the first bin\n  - 'xup' - upper edge of the last bin\n") );
        bp::scope Histogram1D_scope( Histogram1D_exposer );
        Histogram1D_exposer.def( bp::init< int, std::vector< double > const & >(( bp::arg("nbinsx"), bp::arg("xbins") ), "Constructor for variable bin size histograms. @param nbinsx number of bins @param xbins Array of size nbins+1 containing low-edges for each bin and upper edge of last bin. \n\n:Parameters:\n  - 'nbinsx' - number of bins\n  - 'xbins' - Array of size nbins+1 containing low-edges for each\n") );
        Histogram1D_exposer.def( bp::init< IAxis const & >(( bp::arg("axis") ), "Constructor for 1D histogram with custom axis.") );
        Histogram1D_exposer.def( bp::init< OutputData< double > const & >(( bp::arg("data") ), "Constructor for 1D histograms from basic OutputData object.") );
        { //::Histogram1D::fill
        
            typedef int ( ::Histogram1D::*fill_function_type)( double,double ) ;
            
            Histogram1D_exposer.def( 
                "fill"
                , fill_function_type( &::Histogram1D::fill )
                , ( bp::arg("x"), bp::arg("weight")=1.0e+0 )
                , "Increment bin with abscissa x with a weight." );
        
        }
        { //::Histogram1D::getBinCenters
        
            typedef ::std::vector< double > ( ::Histogram1D::*getBinCenters_function_type)(  ) const;
            
            Histogram1D_exposer.def( 
                "getBinCenters"
                , getBinCenters_function_type( &::Histogram1D::getBinCenters )
                , "Increment bin with abscissa x with a weight." );
        
        }
        { //::Histogram1D::getBinValues
        
            typedef ::std::vector< double > ( ::Histogram1D::*getBinValues_function_type)(  ) const;
            
            Histogram1D_exposer.def( 
                "getBinValues"
                , getBinValues_function_type( &::Histogram1D::getBinValues ) );
        
        }
        { //::Histogram1D::getRank
        
            typedef ::std::size_t ( ::Histogram1D::*getRank_function_type)(  ) const;
            typedef ::std::size_t ( Histogram1D_wrapper::*default_getRank_function_type)(  ) const;
            
            Histogram1D_exposer.def( 
                "getRank"
                , getRank_function_type(&::Histogram1D::getRank)
                , default_getRank_function_type(&Histogram1D_wrapper::default_getRank) );
        
        }
        { //::IHistogram::getTotalNumberOfBins
        
            typedef ::std::size_t ( ::IHistogram::*getTotalNumberOfBins_function_type)(  ) const;
            typedef ::std::size_t ( Histogram1D_wrapper::*default_getTotalNumberOfBins_function_type)(  ) const;
            
            Histogram1D_exposer.def( 
                "getTotalNumberOfBins"
                , getTotalNumberOfBins_function_type(&::IHistogram::getTotalNumberOfBins)
                , default_getTotalNumberOfBins_function_type(&Histogram1D_wrapper::default_getTotalNumberOfBins) );
        
        }
        { //::IHistogram::getXaxis
        
            typedef ::IAxis const * ( ::IHistogram::*getXaxis_function_type)(  ) const;
            typedef ::IAxis const * ( Histogram1D_wrapper::*default_getXaxis_function_type)(  ) const;
            
            Histogram1D_exposer.def( 
                "getXaxis"
                , getXaxis_function_type(&::IHistogram::getXaxis)
                , default_getXaxis_function_type(&Histogram1D_wrapper::default_getXaxis)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::IHistogram::getXaxisValue
        
            typedef double ( ::IHistogram::*getXaxisValue_function_type)( ::std::size_t ) ;
            typedef double ( Histogram1D_wrapper::*default_getXaxisValue_function_type)( ::std::size_t ) ;
            
            Histogram1D_exposer.def( 
                "getXaxisValue"
                , getXaxisValue_function_type(&::IHistogram::getXaxisValue)
                , default_getXaxisValue_function_type(&Histogram1D_wrapper::default_getXaxisValue)
                , ( bp::arg("binGlobalIndex") ) );
        
        }
        { //::IHistogram::getYaxis
        
            typedef ::IAxis const * ( ::IHistogram::*getYaxis_function_type)(  ) const;
            typedef ::IAxis const * ( Histogram1D_wrapper::*default_getYaxis_function_type)(  ) const;
            
            Histogram1D_exposer.def( 
                "getYaxis"
                , getYaxis_function_type(&::IHistogram::getYaxis)
                , default_getYaxis_function_type(&Histogram1D_wrapper::default_getYaxis)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::IHistogram::getYaxisValue
        
            typedef double ( ::IHistogram::*getYaxisValue_function_type)( ::std::size_t ) ;
            typedef double ( Histogram1D_wrapper::*default_getYaxisValue_function_type)( ::std::size_t ) ;
            
            Histogram1D_exposer.def( 
                "getYaxisValue"
                , getYaxisValue_function_type(&::IHistogram::getYaxisValue)
                , default_getYaxisValue_function_type(&Histogram1D_wrapper::default_getYaxisValue)
                , ( bp::arg("binGlobalIndex") ) );
        
        }
        { //::IHistogram::reset
        
            typedef void ( ::IHistogram::*reset_function_type)(  ) ;
            typedef void ( Histogram1D_wrapper::*default_reset_function_type)(  ) ;
            
            Histogram1D_exposer.def( 
                "reset"
                , reset_function_type(&::IHistogram::reset)
                , default_reset_function_type(&Histogram1D_wrapper::default_reset) );
        
        }
    }

}
