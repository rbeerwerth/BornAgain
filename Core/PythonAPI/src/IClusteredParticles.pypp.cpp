// This file has been generated by Py++.

// BornAgain: simulate and fit scattering at grazing incidence 
//! @brief automatically generated boost::python code for PythonCoreAPI  

#include "Macros.h"
GCC_DIAG_OFF(unused-parameter);
GCC_DIAG_OFF(missing-field-initializers);
#include "boost/python.hpp"
GCC_DIAG_ON(unused-parameter);
GCC_DIAG_ON(missing-field-initializers);
#include "__call_policies.pypp.hpp"
#include "__convenience.pypp.hpp"
#include "PythonCoreList.h"
#include "IClusteredParticles.pypp.h"

namespace bp = boost::python;

struct IClusteredParticles_wrapper : IClusteredParticles, bp::wrapper< IClusteredParticles > {

    IClusteredParticles_wrapper( )
    : IClusteredParticles( )
      , bp::wrapper< IClusteredParticles >(){
        // null constructor
    
    }

    virtual ::IClusteredParticles * clone(  ) const  {
        if( bp::override func_clone = this->get_override( "clone" ) )
            return func_clone(  );
        else{
            return this->IClusteredParticles::clone(  );
        }
    }
    
    ::IClusteredParticles * default_clone(  ) const  {
        return IClusteredParticles::clone( );
    }

    virtual ::IClusteredParticles * cloneInvertB(  ) const  {
        if( bp::override func_cloneInvertB = this->get_override( "cloneInvertB" ) )
            return func_cloneInvertB(  );
        else{
            return this->IClusteredParticles::cloneInvertB(  );
        }
    }
    
    ::IClusteredParticles * default_cloneInvertB(  ) const  {
        return IClusteredParticles::cloneInvertB( );
    }

    virtual void setAmbientMaterial( ::IMaterial const * p_ambient_material ){
        bp::override func_setAmbientMaterial = this->get_override( "setAmbientMaterial" );
        func_setAmbientMaterial( boost::python::ptr(p_ambient_material) );
    }

    virtual bool areParametersChanged(  ) {
        if( bp::override func_areParametersChanged = this->get_override( "areParametersChanged" ) )
            return func_areParametersChanged(  );
        else{
            return this->IParameterized::areParametersChanged(  );
        }
    }
    
    bool default_areParametersChanged(  ) {
        return IParameterized::areParametersChanged( );
    }

    virtual void clearParameterPool(  ) {
        if( bp::override func_clearParameterPool = this->get_override( "clearParameterPool" ) )
            func_clearParameterPool(  );
        else{
            this->IParameterized::clearParameterPool(  );
        }
    }
    
    void default_clearParameterPool(  ) {
        IParameterized::clearParameterPool( );
    }

    virtual ::ParameterPool * createParameterTree(  ) const  {
        if( bp::override func_createParameterTree = this->get_override( "createParameterTree" ) )
            return func_createParameterTree(  );
        else{
            return this->IParameterized::createParameterTree(  );
        }
    }
    
    ::ParameterPool * default_createParameterTree(  ) const  {
        return IParameterized::createParameterTree( );
    }

    virtual void printParameters(  ) const  {
        if( bp::override func_printParameters = this->get_override( "printParameters" ) )
            func_printParameters(  );
        else{
            this->IParameterized::printParameters(  );
        }
    }
    
    void default_printParameters(  ) const  {
        IParameterized::printParameters( );
    }

    virtual void registerParameter( ::std::string const & name, double * parpointer ) {
        namespace bpl = boost::python;
        if( bpl::override func_registerParameter = this->get_override( "registerParameter" ) ){
            bpl::object py_result = bpl::call<bpl::object>( func_registerParameter.ptr(), name, parpointer );
        }
        else{
            IParameterized::registerParameter( name, parpointer );
        }
    }
    
    static void default_registerParameter( ::IParameterized & inst, ::std::string const & name, long unsigned int parpointer ){
        if( dynamic_cast< IClusteredParticles_wrapper * >( boost::addressof( inst ) ) ){
            inst.::IParameterized::registerParameter(name, reinterpret_cast< double * >( parpointer ));
        }
        else{
            inst.registerParameter(name, reinterpret_cast< double * >( parpointer ));
        }
    }

    virtual bool setParameterValue( ::std::string const & name, double value ) {
        if( bp::override func_setParameterValue = this->get_override( "setParameterValue" ) )
            return func_setParameterValue( name, value );
        else{
            return this->IParameterized::setParameterValue( name, value );
        }
    }
    
    bool default_setParameterValue( ::std::string const & name, double value ) {
        return IParameterized::setParameterValue( name, value );
    }

    virtual void setParametersAreChanged(  ) {
        if( bp::override func_setParametersAreChanged = this->get_override( "setParametersAreChanged" ) )
            func_setParametersAreChanged(  );
        else{
            this->IParameterized::setParametersAreChanged(  );
        }
    }
    
    void default_setParametersAreChanged(  ) {
        IParameterized::setParametersAreChanged( );
    }

};

void register_IClusteredParticles_class(){

    { //::IClusteredParticles
        typedef bp::class_< IClusteredParticles_wrapper, bp::bases< ICompositeSample >, boost::noncopyable > IClusteredParticles_exposer_t;
        IClusteredParticles_exposer_t IClusteredParticles_exposer = IClusteredParticles_exposer_t( "IClusteredParticles", bp::init< >() );
        bp::scope IClusteredParticles_scope( IClusteredParticles_exposer );
        { //::IClusteredParticles::clone
        
            typedef ::IClusteredParticles * ( ::IClusteredParticles::*clone_function_type )(  ) const;
            typedef ::IClusteredParticles * ( IClusteredParticles_wrapper::*default_clone_function_type )(  ) const;
            
            IClusteredParticles_exposer.def( 
                "clone"
                , clone_function_type(&::IClusteredParticles::clone)
                , default_clone_function_type(&IClusteredParticles_wrapper::default_clone)
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::IClusteredParticles::cloneInvertB
        
            typedef ::IClusteredParticles * ( ::IClusteredParticles::*cloneInvertB_function_type )(  ) const;
            typedef ::IClusteredParticles * ( IClusteredParticles_wrapper::*default_cloneInvertB_function_type )(  ) const;
            
            IClusteredParticles_exposer.def( 
                "cloneInvertB"
                , cloneInvertB_function_type(&::IClusteredParticles::cloneInvertB)
                , default_cloneInvertB_function_type(&IClusteredParticles_wrapper::default_cloneInvertB)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::IClusteredParticles::setAmbientMaterial
        
            typedef void ( ::IClusteredParticles::*setAmbientMaterial_function_type )( ::IMaterial const * ) ;
            
            IClusteredParticles_exposer.def( 
                "setAmbientMaterial"
                , bp::pure_virtual( setAmbientMaterial_function_type(&::IClusteredParticles::setAmbientMaterial) )
                , ( bp::arg("p_ambient_material") ) );
        
        }
        { //::IParameterized::areParametersChanged
        
            typedef bool ( ::IParameterized::*areParametersChanged_function_type )(  ) ;
            typedef bool ( IClusteredParticles_wrapper::*default_areParametersChanged_function_type )(  ) ;
            
            IClusteredParticles_exposer.def( 
                "areParametersChanged"
                , areParametersChanged_function_type(&::IParameterized::areParametersChanged)
                , default_areParametersChanged_function_type(&IClusteredParticles_wrapper::default_areParametersChanged) );
        
        }
        { //::IParameterized::clearParameterPool
        
            typedef void ( ::IParameterized::*clearParameterPool_function_type )(  ) ;
            typedef void ( IClusteredParticles_wrapper::*default_clearParameterPool_function_type )(  ) ;
            
            IClusteredParticles_exposer.def( 
                "clearParameterPool"
                , clearParameterPool_function_type(&::IParameterized::clearParameterPool)
                , default_clearParameterPool_function_type(&IClusteredParticles_wrapper::default_clearParameterPool) );
        
        }
        { //::IParameterized::createParameterTree
        
            typedef ::ParameterPool * ( ::IParameterized::*createParameterTree_function_type )(  ) const;
            typedef ::ParameterPool * ( IClusteredParticles_wrapper::*default_createParameterTree_function_type )(  ) const;
            
            IClusteredParticles_exposer.def( 
                "createParameterTree"
                , createParameterTree_function_type(&::IParameterized::createParameterTree)
                , default_createParameterTree_function_type(&IClusteredParticles_wrapper::default_createParameterTree)
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::IParameterized::printParameters
        
            typedef void ( ::IParameterized::*printParameters_function_type )(  ) const;
            typedef void ( IClusteredParticles_wrapper::*default_printParameters_function_type )(  ) const;
            
            IClusteredParticles_exposer.def( 
                "printParameters"
                , printParameters_function_type(&::IParameterized::printParameters)
                , default_printParameters_function_type(&IClusteredParticles_wrapper::default_printParameters) );
        
        }
        { //::IParameterized::registerParameter
        
            typedef void ( *default_registerParameter_function_type )( ::IParameterized &,::std::string const &,long unsigned int );
            
            IClusteredParticles_exposer.def( 
                "registerParameter"
                , default_registerParameter_function_type( &IClusteredParticles_wrapper::default_registerParameter )
                , ( bp::arg("inst"), bp::arg("name"), bp::arg("parpointer") ) );
        
        }
        { //::IParameterized::setParameterValue
        
            typedef bool ( ::IParameterized::*setParameterValue_function_type )( ::std::string const &,double ) ;
            typedef bool ( IClusteredParticles_wrapper::*default_setParameterValue_function_type )( ::std::string const &,double ) ;
            
            IClusteredParticles_exposer.def( 
                "setParameterValue"
                , setParameterValue_function_type(&::IParameterized::setParameterValue)
                , default_setParameterValue_function_type(&IClusteredParticles_wrapper::default_setParameterValue)
                , ( bp::arg("name"), bp::arg("value") ) );
        
        }
        { //::IParameterized::setParametersAreChanged
        
            typedef void ( ::IParameterized::*setParametersAreChanged_function_type )(  ) ;
            typedef void ( IClusteredParticles_wrapper::*default_setParametersAreChanged_function_type )(  ) ;
            
            IClusteredParticles_exposer.def( 
                "setParametersAreChanged"
                , setParametersAreChanged_function_type(&::IParameterized::setParametersAreChanged)
                , default_setParametersAreChanged_function_type(&IClusteredParticles_wrapper::default_setParametersAreChanged) );
        
        }
    }

}
