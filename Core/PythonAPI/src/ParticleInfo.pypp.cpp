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
#include "__call_policies.pypp.hpp"
#include "__convenience.pypp.hpp"
#include "PythonCoreList.h"
#include "ParticleInfo.pypp.h"

namespace bp = boost::python;

struct ParticleInfo_wrapper : ParticleInfo, bp::wrapper< ParticleInfo > {

    ParticleInfo_wrapper(::IParticle const & p_particle, double depth=0.0, double abundance=1.0e+0 )
    : ParticleInfo( boost::ref(p_particle), depth, abundance )
      , bp::wrapper< ParticleInfo >(){
        // constructor
    m_pyobj = 0;
    }

    ParticleInfo_wrapper(::IParticle const & p_particle, ::kvector_t position, double abundance=1.0e+0 )
    : ParticleInfo( boost::ref(p_particle), position, abundance )
      , bp::wrapper< ParticleInfo >(){
        // constructor
    m_pyobj = 0;
    }

    virtual ::ParticleInfo * clone(  ) const  {
        if( bp::override func_clone = this->get_override( "clone" ) )
            return func_clone(  );
        else{
            return this->ParticleInfo::clone(  );
        }
    }
    
    ::ParticleInfo * default_clone(  ) const  {
        return ParticleInfo::clone( );
    }

    virtual ::ParticleInfo * cloneInvertB(  ) const  {
        if( bp::override func_cloneInvertB = this->get_override( "cloneInvertB" ) )
            return func_cloneInvertB(  );
        else{
            return this->ParticleInfo::cloneInvertB(  );
        }
    }
    
    ::ParticleInfo * default_cloneInvertB(  ) const  {
        return ParticleInfo::cloneInvertB( );
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

    virtual bool containsMagneticMaterial(  ) const  {
        if( bp::override func_containsMagneticMaterial = this->get_override( "containsMagneticMaterial" ) )
            return func_containsMagneticMaterial(  );
        else{
            return this->ISample::containsMagneticMaterial(  );
        }
    }
    
    bool default_containsMagneticMaterial(  ) const  {
        return ISample::containsMagneticMaterial( );
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

    virtual ::ICompositeSample * getCompositeSample(  ) {
        if( bp::override func_getCompositeSample = this->get_override( "getCompositeSample" ) )
            return func_getCompositeSample(  );
        else{
            return this->ICompositeSample::getCompositeSample(  );
        }
    }
    
    ::ICompositeSample * default_getCompositeSample(  ) {
        return ICompositeSample::getCompositeSample( );
    }

    virtual ::ICompositeSample const * getCompositeSample(  ) const  {
        if( bp::override func_getCompositeSample = this->get_override( "getCompositeSample" ) )
            return func_getCompositeSample(  );
        else{
            return this->ICompositeSample::getCompositeSample(  );
        }
    }
    
    ::ICompositeSample const * default_getCompositeSample(  ) const  {
        return ICompositeSample::getCompositeSample( );
    }

    virtual bool preprocess(  ) {
        if( bp::override func_preprocess = this->get_override( "preprocess" ) )
            return func_preprocess(  );
        else{
            return this->ISample::preprocess(  );
        }
    }
    
    bool default_preprocess(  ) {
        return ISample::preprocess( );
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

    virtual void printSampleTree(  ) {
        if( bp::override func_printSampleTree = this->get_override( "printSampleTree" ) )
            func_printSampleTree(  );
        else{
            this->ISample::printSampleTree(  );
        }
    }
    
    void default_printSampleTree(  ) {
        ISample::printSampleTree( );
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
        if( dynamic_cast< ParticleInfo_wrapper * >( boost::addressof( inst ) ) ){
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

    virtual ::std::size_t size(  ) const  {
        if( bp::override func_size = this->get_override( "size" ) )
            return func_size(  );
        else{
            return this->ICompositeSample::size(  );
        }
    }
    
    ::std::size_t default_size(  ) const  {
        return ICompositeSample::size( );
    }

    virtual void transferToCPP(  ) {
        
        if( !this->m_pyobj) {
            this->m_pyobj = boost::python::detail::wrapper_base_::get_owner(*this);
            Py_INCREF(this->m_pyobj);
        }
        
        if( bp::override func_transferToCPP = this->get_override( "transferToCPP" ) )
            func_transferToCPP(  );
        else{
            this->ICloneable::transferToCPP(  );
        }
    }
    
    void default_transferToCPP(  ) {
        
        if( !this->m_pyobj) {
            this->m_pyobj = boost::python::detail::wrapper_base_::get_owner(*this);
            Py_INCREF(this->m_pyobj);
        }
        
        ICloneable::transferToCPP( );
    }

    PyObject* m_pyobj;

};

void register_ParticleInfo_class(){

    { //::ParticleInfo
        typedef bp::class_< ParticleInfo_wrapper, bp::bases< ICompositeSample >, std::auto_ptr< ParticleInfo_wrapper >, boost::noncopyable > ParticleInfo_exposer_t;
        ParticleInfo_exposer_t ParticleInfo_exposer = ParticleInfo_exposer_t( "ParticleInfo", bp::init< IParticle const &, bp::optional< double, double > >(( bp::arg("p_particle"), bp::arg("depth")=0.0, bp::arg("abundance")=1.0e+0 )) );
        bp::scope ParticleInfo_scope( ParticleInfo_exposer );
        ParticleInfo_exposer.def( bp::init< IParticle const &, kvector_t, bp::optional< double > >(( bp::arg("p_particle"), bp::arg("position"), bp::arg("abundance")=1.0e+0 )) );
        { //::ParticleInfo::clone
        
            typedef ::ParticleInfo * ( ::ParticleInfo::*clone_function_type)(  ) const;
            typedef ::ParticleInfo * ( ParticleInfo_wrapper::*default_clone_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "clone"
                , clone_function_type(&::ParticleInfo::clone)
                , default_clone_function_type(&ParticleInfo_wrapper::default_clone)
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::ParticleInfo::cloneInvertB
        
            typedef ::ParticleInfo * ( ::ParticleInfo::*cloneInvertB_function_type)(  ) const;
            typedef ::ParticleInfo * ( ParticleInfo_wrapper::*default_cloneInvertB_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "cloneInvertB"
                , cloneInvertB_function_type(&::ParticleInfo::cloneInvertB)
                , default_cloneInvertB_function_type(&ParticleInfo_wrapper::default_cloneInvertB)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ParticleInfo::getAbundance
        
            typedef double ( ::ParticleInfo::*getAbundance_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "getAbundance"
                , getAbundance_function_type( &::ParticleInfo::getAbundance ) );
        
        }
        { //::ParticleInfo::getDepth
        
            typedef double ( ::ParticleInfo::*getDepth_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "getDepth"
                , getDepth_function_type( &::ParticleInfo::getDepth ) );
        
        }
        { //::ParticleInfo::getParticle
        
            typedef ::IParticle const * ( ::ParticleInfo::*getParticle_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "getParticle"
                , getParticle_function_type( &::ParticleInfo::getParticle )
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ParticleInfo::getPosition
        
            typedef ::kvector_t ( ::ParticleInfo::*getPosition_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "getPosition"
                , getPosition_function_type( &::ParticleInfo::getPosition ) );
        
        }
        { //::ParticleInfo::setAbundance
        
            typedef void ( ::ParticleInfo::*setAbundance_function_type)( double ) ;
            
            ParticleInfo_exposer.def( 
                "setAbundance"
                , setAbundance_function_type( &::ParticleInfo::setAbundance )
                , ( bp::arg("abundance") ) );
        
        }
        { //::ParticleInfo::setAmbientMaterial
        
            typedef void ( ::ParticleInfo::*setAmbientMaterial_function_type)( ::IMaterial const & ) ;
            
            ParticleInfo_exposer.def( 
                "setAmbientMaterial"
                , setAmbientMaterial_function_type( &::ParticleInfo::setAmbientMaterial )
                , ( bp::arg("material") ) );
        
        }
        { //::ParticleInfo::setPosition
        
            typedef void ( ::ParticleInfo::*setPosition_function_type)( ::kvector_t ) ;
            
            ParticleInfo_exposer.def( 
                "setPosition"
                , setPosition_function_type( &::ParticleInfo::setPosition )
                , ( bp::arg("position") ) );
        
        }
        { //::IParameterized::areParametersChanged
        
            typedef bool ( ::IParameterized::*areParametersChanged_function_type)(  ) ;
            typedef bool ( ParticleInfo_wrapper::*default_areParametersChanged_function_type)(  ) ;
            
            ParticleInfo_exposer.def( 
                "areParametersChanged"
                , areParametersChanged_function_type(&::IParameterized::areParametersChanged)
                , default_areParametersChanged_function_type(&ParticleInfo_wrapper::default_areParametersChanged) );
        
        }
        { //::IParameterized::clearParameterPool
        
            typedef void ( ::IParameterized::*clearParameterPool_function_type)(  ) ;
            typedef void ( ParticleInfo_wrapper::*default_clearParameterPool_function_type)(  ) ;
            
            ParticleInfo_exposer.def( 
                "clearParameterPool"
                , clearParameterPool_function_type(&::IParameterized::clearParameterPool)
                , default_clearParameterPool_function_type(&ParticleInfo_wrapper::default_clearParameterPool) );
        
        }
        { //::ISample::containsMagneticMaterial
        
            typedef bool ( ::ISample::*containsMagneticMaterial_function_type)(  ) const;
            typedef bool ( ParticleInfo_wrapper::*default_containsMagneticMaterial_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "containsMagneticMaterial"
                , containsMagneticMaterial_function_type(&::ISample::containsMagneticMaterial)
                , default_containsMagneticMaterial_function_type(&ParticleInfo_wrapper::default_containsMagneticMaterial) );
        
        }
        { //::IParameterized::createParameterTree
        
            typedef ::ParameterPool * ( ::IParameterized::*createParameterTree_function_type)(  ) const;
            typedef ::ParameterPool * ( ParticleInfo_wrapper::*default_createParameterTree_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "createParameterTree"
                , createParameterTree_function_type(&::IParameterized::createParameterTree)
                , default_createParameterTree_function_type(&ParticleInfo_wrapper::default_createParameterTree)
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::ICompositeSample::getCompositeSample
        
            typedef ::ICompositeSample * ( ::ICompositeSample::*getCompositeSample_function_type)(  ) ;
            typedef ::ICompositeSample * ( ParticleInfo_wrapper::*default_getCompositeSample_function_type)(  ) ;
            
            ParticleInfo_exposer.def( 
                "getCompositeSample"
                , getCompositeSample_function_type(&::ICompositeSample::getCompositeSample)
                , default_getCompositeSample_function_type(&ParticleInfo_wrapper::default_getCompositeSample)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ICompositeSample::getCompositeSample
        
            typedef ::ICompositeSample const * ( ::ICompositeSample::*getCompositeSample_function_type)(  ) const;
            typedef ::ICompositeSample const * ( ParticleInfo_wrapper::*default_getCompositeSample_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "getCompositeSample"
                , getCompositeSample_function_type(&::ICompositeSample::getCompositeSample)
                , default_getCompositeSample_function_type(&ParticleInfo_wrapper::default_getCompositeSample)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ISample::preprocess
        
            typedef bool ( ::ISample::*preprocess_function_type)(  ) ;
            typedef bool ( ParticleInfo_wrapper::*default_preprocess_function_type)(  ) ;
            
            ParticleInfo_exposer.def( 
                "preprocess"
                , preprocess_function_type(&::ISample::preprocess)
                , default_preprocess_function_type(&ParticleInfo_wrapper::default_preprocess) );
        
        }
        { //::IParameterized::printParameters
        
            typedef void ( ::IParameterized::*printParameters_function_type)(  ) const;
            typedef void ( ParticleInfo_wrapper::*default_printParameters_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "printParameters"
                , printParameters_function_type(&::IParameterized::printParameters)
                , default_printParameters_function_type(&ParticleInfo_wrapper::default_printParameters) );
        
        }
        { //::ISample::printSampleTree
        
            typedef void ( ::ISample::*printSampleTree_function_type)(  ) ;
            typedef void ( ParticleInfo_wrapper::*default_printSampleTree_function_type)(  ) ;
            
            ParticleInfo_exposer.def( 
                "printSampleTree"
                , printSampleTree_function_type(&::ISample::printSampleTree)
                , default_printSampleTree_function_type(&ParticleInfo_wrapper::default_printSampleTree) );
        
        }
        { //::IParameterized::registerParameter
        
            typedef void ( *default_registerParameter_function_type )( ::IParameterized &,::std::string const &,long unsigned int );
            
            ParticleInfo_exposer.def( 
                "registerParameter"
                , default_registerParameter_function_type( &ParticleInfo_wrapper::default_registerParameter )
                , ( bp::arg("inst"), bp::arg("name"), bp::arg("parpointer") ) );
        
        }
        { //::IParameterized::setParameterValue
        
            typedef bool ( ::IParameterized::*setParameterValue_function_type)( ::std::string const &,double ) ;
            typedef bool ( ParticleInfo_wrapper::*default_setParameterValue_function_type)( ::std::string const &,double ) ;
            
            ParticleInfo_exposer.def( 
                "setParameterValue"
                , setParameterValue_function_type(&::IParameterized::setParameterValue)
                , default_setParameterValue_function_type(&ParticleInfo_wrapper::default_setParameterValue)
                , ( bp::arg("name"), bp::arg("value") ) );
        
        }
        { //::IParameterized::setParametersAreChanged
        
            typedef void ( ::IParameterized::*setParametersAreChanged_function_type)(  ) ;
            typedef void ( ParticleInfo_wrapper::*default_setParametersAreChanged_function_type)(  ) ;
            
            ParticleInfo_exposer.def( 
                "setParametersAreChanged"
                , setParametersAreChanged_function_type(&::IParameterized::setParametersAreChanged)
                , default_setParametersAreChanged_function_type(&ParticleInfo_wrapper::default_setParametersAreChanged) );
        
        }
        { //::ICompositeSample::size
        
            typedef ::std::size_t ( ::ICompositeSample::*size_function_type)(  ) const;
            typedef ::std::size_t ( ParticleInfo_wrapper::*default_size_function_type)(  ) const;
            
            ParticleInfo_exposer.def( 
                "size"
                , size_function_type(&::ICompositeSample::size)
                , default_size_function_type(&ParticleInfo_wrapper::default_size) );
        
        }
        { //::ICloneable::transferToCPP
        
            typedef void ( ::ICloneable::*transferToCPP_function_type)(  ) ;
            typedef void ( ParticleInfo_wrapper::*default_transferToCPP_function_type)(  ) ;
            
            ParticleInfo_exposer.def( 
                "transferToCPP"
                , transferToCPP_function_type(&::ICloneable::transferToCPP)
                , default_transferToCPP_function_type(&ParticleInfo_wrapper::default_transferToCPP) );
        
        }
    }

}
