// This file has been generated by Py++.

// BornAgain: simulate and fit scattering at grazing incidence 
//! @brief automatically generated boost::python code for PythonCoreAPI  

#include "Macros.h"
GCC_DIAG_OFF(unused-parameter);
GCC_DIAG_OFF(missing-field-initializers);
#include "boost/python.hpp"
GCC_DIAG_ON(unused-parameter);
GCC_DIAG_ON(missing-field-initializers);
#include "PythonFitList.h"
#include "FitStrategyAdjustMinimizer.pypp.h"

namespace bp = boost::python;

struct FitStrategyAdjustMinimizer_wrapper : FitStrategyAdjustMinimizer, bp::wrapper< FitStrategyAdjustMinimizer > {

    FitStrategyAdjustMinimizer_wrapper(FitStrategyAdjustMinimizer const & arg )
    : FitStrategyAdjustMinimizer( arg )
      , bp::wrapper< FitStrategyAdjustMinimizer >(){
        // copy constructor
        
    }

    FitStrategyAdjustMinimizer_wrapper( )
    : FitStrategyAdjustMinimizer( )
      , bp::wrapper< FitStrategyAdjustMinimizer >(){
        // null constructor
    
    }

    virtual ::FitStrategyAdjustMinimizer * clone(  ) const  {
        if( bp::override func_clone = this->get_override( "clone" ) )
            return func_clone(  );
        else
            return this->FitStrategyAdjustMinimizer::clone(  );
    }
    
    
    ::FitStrategyAdjustMinimizer * default_clone(  ) const  {
        return FitStrategyAdjustMinimizer::clone( );
    }

    virtual void execute(  ) {
        if( bp::override func_execute = this->get_override( "execute" ) )
            func_execute(  );
        else
            this->FitStrategyAdjustMinimizer::execute(  );
    }
    
    
    void default_execute(  ) {
        FitStrategyAdjustMinimizer::execute( );
    }

};

void register_FitStrategyAdjustMinimizer_class(){

    { //::FitStrategyAdjustMinimizer
        typedef bp::class_< FitStrategyAdjustMinimizer_wrapper, bp::bases< IFitStrategy > > FitStrategyAdjustMinimizer_exposer_t;
        FitStrategyAdjustMinimizer_exposer_t FitStrategyAdjustMinimizer_exposer = FitStrategyAdjustMinimizer_exposer_t( "FitStrategyAdjustMinimizer", bp::init< >() );
        bp::scope FitStrategyAdjustMinimizer_scope( FitStrategyAdjustMinimizer_exposer );
        { //::FitStrategyAdjustMinimizer::clone
        
            typedef ::FitStrategyAdjustMinimizer * ( ::FitStrategyAdjustMinimizer::*clone_function_type )(  ) const;
            typedef ::FitStrategyAdjustMinimizer * ( FitStrategyAdjustMinimizer_wrapper::*default_clone_function_type )(  ) const;
            
            FitStrategyAdjustMinimizer_exposer.def( 
                "clone"
                , clone_function_type(&::FitStrategyAdjustMinimizer::clone)
                , default_clone_function_type(&FitStrategyAdjustMinimizer_wrapper::default_clone)
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::FitStrategyAdjustMinimizer::execute
        
            typedef void ( ::FitStrategyAdjustMinimizer::*execute_function_type )(  ) ;
            typedef void ( FitStrategyAdjustMinimizer_wrapper::*default_execute_function_type )(  ) ;
            
            FitStrategyAdjustMinimizer_exposer.def( 
                "execute"
                , execute_function_type(&::FitStrategyAdjustMinimizer::execute)
                , default_execute_function_type(&FitStrategyAdjustMinimizer_wrapper::default_execute) );
        
        }
        { //::FitStrategyAdjustMinimizer::getMinimizer
        
            typedef ::IMinimizer * ( ::FitStrategyAdjustMinimizer::*getMinimizer_function_type )(  ) ;
            
            FitStrategyAdjustMinimizer_exposer.def( 
                "getMinimizer"
                , getMinimizer_function_type( &::FitStrategyAdjustMinimizer::getMinimizer )
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::FitStrategyAdjustMinimizer::setMinimizer
        
            typedef void ( ::FitStrategyAdjustMinimizer::*setMinimizer_function_type )( ::IMinimizer * ) ;
            
            FitStrategyAdjustMinimizer_exposer.def( 
                "setMinimizer"
                , setMinimizer_function_type( &::FitStrategyAdjustMinimizer::setMinimizer )
                , ( bp::arg("minimizer") ) );
        
        }
    }

}
