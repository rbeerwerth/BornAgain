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
#include "Particle.pypp.h"

namespace bp = boost::python;

struct Particle_wrapper : Particle, bp::wrapper< Particle > {

    Particle_wrapper( )
    : Particle( )
      , bp::wrapper< Particle >(){
        // null constructor
    m_pyobj = 0;
    }

    Particle_wrapper(::IMaterial const & p_material )
    : Particle( boost::ref(p_material) )
      , bp::wrapper< Particle >(){
        // constructor
    m_pyobj = 0;
    }

    Particle_wrapper(::IMaterial const & p_material, ::IFormFactor const & form_factor )
    : Particle( boost::ref(p_material), boost::ref(form_factor) )
      , bp::wrapper< Particle >(){
        // constructor
    m_pyobj = 0;
    }

    Particle_wrapper(::IMaterial const & p_material, ::IFormFactor const & form_factor, ::IRotation const & rotation )
    : Particle( boost::ref(p_material), boost::ref(form_factor), boost::ref(rotation) )
      , bp::wrapper< Particle >(){
        // constructor
    m_pyobj = 0;
    }

    virtual ::Particle * clone(  ) const  {
        if( bp::override func_clone = this->get_override( "clone" ) )
            return func_clone(  );
        else{
            return this->Particle::clone(  );
        }
    }
    
    ::Particle * default_clone(  ) const  {
        return Particle::clone( );
    }

    virtual ::Particle * cloneInvertB(  ) const  {
        if( bp::override func_cloneInvertB = this->get_override( "cloneInvertB" ) )
            return func_cloneInvertB(  );
        else{
            return this->Particle::cloneInvertB(  );
        }
    }
    
    ::Particle * default_cloneInvertB(  ) const  {
        return Particle::cloneInvertB( );
    }

    virtual ::IMaterial const * getAmbientMaterial(  ) const  {
        if( bp::override func_getAmbientMaterial = this->get_override( "getAmbientMaterial" ) )
            return func_getAmbientMaterial(  );
        else{
            return this->Particle::getAmbientMaterial(  );
        }
    }
    
    ::IMaterial const * default_getAmbientMaterial(  ) const  {
        return Particle::getAmbientMaterial( );
    }

    virtual ::IMaterial const * getMaterial(  ) const  {
        if( bp::override func_getMaterial = this->get_override( "getMaterial" ) )
            return func_getMaterial(  );
        else{
            return this->Particle::getMaterial(  );
        }
    }
    
    ::IMaterial const * default_getMaterial(  ) const  {
        return Particle::getMaterial( );
    }

    virtual ::complex_t getRefractiveIndex(  ) const  {
        if( bp::override func_getRefractiveIndex = this->get_override( "getRefractiveIndex" ) )
            return func_getRefractiveIndex(  );
        else{
            return this->Particle::getRefractiveIndex(  );
        }
    }
    
    ::complex_t default_getRefractiveIndex(  ) const  {
        return Particle::getRefractiveIndex( );
    }

    virtual void setAmbientMaterial( ::IMaterial const & material ) {
        if( bp::override func_setAmbientMaterial = this->get_override( "setAmbientMaterial" ) )
            func_setAmbientMaterial( boost::ref(material) );
        else{
            this->Particle::setAmbientMaterial( boost::ref(material) );
        }
    }
    
    void default_setAmbientMaterial( ::IMaterial const & material ) {
        Particle::setAmbientMaterial( boost::ref(material) );
    }

    virtual void setMaterial( ::IMaterial const & material ) {
        if( bp::override func_setMaterial = this->get_override( "setMaterial" ) )
            func_setMaterial( boost::ref(material) );
        else{
            this->Particle::setMaterial( boost::ref(material) );
        }
    }
    
    void default_setMaterial( ::IMaterial const & material ) {
        Particle::setMaterial( boost::ref(material) );
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

    virtual void registerParameter( ::std::string const & name, double * parpointer, ::AttLimits const & limits=AttLimits::limitless( ) ) {
        namespace bpl = boost::python;
        if( bpl::override func_registerParameter = this->get_override( "registerParameter" ) ){
            bpl::object py_result = bpl::call<bpl::object>( func_registerParameter.ptr(), name, parpointer, limits );
        }
        else{
            IParameterized::registerParameter( name, parpointer, boost::ref(limits) );
        }
    }
    
    static void default_registerParameter( ::IParameterized & inst, ::std::string const & name, long unsigned int parpointer, ::AttLimits const & limits=AttLimits::limitless( ) ){
        if( dynamic_cast< Particle_wrapper * >( boost::addressof( inst ) ) ){
            inst.::IParameterized::registerParameter(name, reinterpret_cast< double * >( parpointer ), limits);
        }
        else{
            inst.registerParameter(name, reinterpret_cast< double * >( parpointer ), limits);
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

void register_Particle_class(){

    { //::Particle
        typedef bp::class_< Particle_wrapper, bp::bases< IParticle >, std::auto_ptr< Particle_wrapper >, boost::noncopyable > Particle_exposer_t;
        Particle_exposer_t Particle_exposer = Particle_exposer_t( "Particle", "A particle with a form factor and refractive inde.", bp::init< >() );
        bp::scope Particle_scope( Particle_exposer );
        Particle_exposer.def( bp::init< IMaterial const & >(( bp::arg("p_material") )) );
        Particle_exposer.def( bp::init< IMaterial const &, IFormFactor const & >(( bp::arg("p_material"), bp::arg("form_factor") )) );
        Particle_exposer.def( bp::init< IMaterial const &, IFormFactor const &, IRotation const & >(( bp::arg("p_material"), bp::arg("form_factor"), bp::arg("rotation") )) );
        { //::Particle::clone
        
            typedef ::Particle * ( ::Particle::*clone_function_type)(  ) const;
            typedef ::Particle * ( Particle_wrapper::*default_clone_function_type)(  ) const;
            
            Particle_exposer.def( 
                "clone"
                , clone_function_type(&::Particle::clone)
                , default_clone_function_type(&Particle_wrapper::default_clone)
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::Particle::cloneInvertB
        
            typedef ::Particle * ( ::Particle::*cloneInvertB_function_type)(  ) const;
            typedef ::Particle * ( Particle_wrapper::*default_cloneInvertB_function_type)(  ) const;
            
            Particle_exposer.def( 
                "cloneInvertB"
                , cloneInvertB_function_type(&::Particle::cloneInvertB)
                , default_cloneInvertB_function_type(&Particle_wrapper::default_cloneInvertB)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::Particle::getAmbientMaterial
        
            typedef ::IMaterial const * ( ::Particle::*getAmbientMaterial_function_type)(  ) const;
            typedef ::IMaterial const * ( Particle_wrapper::*default_getAmbientMaterial_function_type)(  ) const;
            
            Particle_exposer.def( 
                "getAmbientMaterial"
                , getAmbientMaterial_function_type(&::Particle::getAmbientMaterial)
                , default_getAmbientMaterial_function_type(&Particle_wrapper::default_getAmbientMaterial)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::Particle::getFormFactor
        
            typedef ::IFormFactor const * ( ::Particle::*getFormFactor_function_type)(  ) const;
            
            Particle_exposer.def( 
                "getFormFactor"
                , getFormFactor_function_type( &::Particle::getFormFactor )
                , bp::return_value_policy< bp::reference_existing_object >()
                , "Returns the form factor." );
        
        }
        { //::Particle::getMaterial
        
            typedef ::IMaterial const * ( ::Particle::*getMaterial_function_type)(  ) const;
            typedef ::IMaterial const * ( Particle_wrapper::*default_getMaterial_function_type)(  ) const;
            
            Particle_exposer.def( 
                "getMaterial"
                , getMaterial_function_type(&::Particle::getMaterial)
                , default_getMaterial_function_type(&Particle_wrapper::default_getMaterial)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::Particle::getRefractiveIndex
        
            typedef ::complex_t ( ::Particle::*getRefractiveIndex_function_type)(  ) const;
            typedef ::complex_t ( Particle_wrapper::*default_getRefractiveIndex_function_type)(  ) const;
            
            Particle_exposer.def( 
                "getRefractiveIndex"
                , getRefractiveIndex_function_type(&::Particle::getRefractiveIndex)
                , default_getRefractiveIndex_function_type(&Particle_wrapper::default_getRefractiveIndex) );
        
        }
        { //::Particle::setAmbientMaterial
        
            typedef void ( ::Particle::*setAmbientMaterial_function_type)( ::IMaterial const & ) ;
            typedef void ( Particle_wrapper::*default_setAmbientMaterial_function_type)( ::IMaterial const & ) ;
            
            Particle_exposer.def( 
                "setAmbientMaterial"
                , setAmbientMaterial_function_type(&::Particle::setAmbientMaterial)
                , default_setAmbientMaterial_function_type(&Particle_wrapper::default_setAmbientMaterial)
                , ( bp::arg("material") ) );
        
        }
        { //::Particle::setFormFactor
        
            typedef void ( ::Particle::*setFormFactor_function_type)( ::IFormFactor const & ) ;
            
            Particle_exposer.def( 
                "setFormFactor"
                , setFormFactor_function_type( &::Particle::setFormFactor )
                , ( bp::arg("form_factor") )
                , "Sets the form factor." );
        
        }
        { //::Particle::setMaterial
        
            typedef void ( ::Particle::*setMaterial_function_type)( ::IMaterial const & ) ;
            typedef void ( Particle_wrapper::*default_setMaterial_function_type)( ::IMaterial const & ) ;
            
            Particle_exposer.def( 
                "setMaterial"
                , setMaterial_function_type(&::Particle::setMaterial)
                , default_setMaterial_function_type(&Particle_wrapper::default_setMaterial)
                , ( bp::arg("material") ) );
        
        }
        { //::IParameterized::areParametersChanged
        
            typedef bool ( ::IParameterized::*areParametersChanged_function_type)(  ) ;
            typedef bool ( Particle_wrapper::*default_areParametersChanged_function_type)(  ) ;
            
            Particle_exposer.def( 
                "areParametersChanged"
                , areParametersChanged_function_type(&::IParameterized::areParametersChanged)
                , default_areParametersChanged_function_type(&Particle_wrapper::default_areParametersChanged) );
        
        }
        { //::IParameterized::clearParameterPool
        
            typedef void ( ::IParameterized::*clearParameterPool_function_type)(  ) ;
            typedef void ( Particle_wrapper::*default_clearParameterPool_function_type)(  ) ;
            
            Particle_exposer.def( 
                "clearParameterPool"
                , clearParameterPool_function_type(&::IParameterized::clearParameterPool)
                , default_clearParameterPool_function_type(&Particle_wrapper::default_clearParameterPool) );
        
        }
        { //::ISample::containsMagneticMaterial
        
            typedef bool ( ::ISample::*containsMagneticMaterial_function_type)(  ) const;
            typedef bool ( Particle_wrapper::*default_containsMagneticMaterial_function_type)(  ) const;
            
            Particle_exposer.def( 
                "containsMagneticMaterial"
                , containsMagneticMaterial_function_type(&::ISample::containsMagneticMaterial)
                , default_containsMagneticMaterial_function_type(&Particle_wrapper::default_containsMagneticMaterial) );
        
        }
        { //::IParameterized::createParameterTree
        
            typedef ::ParameterPool * ( ::IParameterized::*createParameterTree_function_type)(  ) const;
            typedef ::ParameterPool * ( Particle_wrapper::*default_createParameterTree_function_type)(  ) const;
            
            Particle_exposer.def( 
                "createParameterTree"
                , createParameterTree_function_type(&::IParameterized::createParameterTree)
                , default_createParameterTree_function_type(&Particle_wrapper::default_createParameterTree)
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::ICompositeSample::getCompositeSample
        
            typedef ::ICompositeSample * ( ::ICompositeSample::*getCompositeSample_function_type)(  ) ;
            typedef ::ICompositeSample * ( Particle_wrapper::*default_getCompositeSample_function_type)(  ) ;
            
            Particle_exposer.def( 
                "getCompositeSample"
                , getCompositeSample_function_type(&::ICompositeSample::getCompositeSample)
                , default_getCompositeSample_function_type(&Particle_wrapper::default_getCompositeSample)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ICompositeSample::getCompositeSample
        
            typedef ::ICompositeSample const * ( ::ICompositeSample::*getCompositeSample_function_type)(  ) const;
            typedef ::ICompositeSample const * ( Particle_wrapper::*default_getCompositeSample_function_type)(  ) const;
            
            Particle_exposer.def( 
                "getCompositeSample"
                , getCompositeSample_function_type(&::ICompositeSample::getCompositeSample)
                , default_getCompositeSample_function_type(&Particle_wrapper::default_getCompositeSample)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::IParameterized::printParameters
        
            typedef void ( ::IParameterized::*printParameters_function_type)(  ) const;
            typedef void ( Particle_wrapper::*default_printParameters_function_type)(  ) const;
            
            Particle_exposer.def( 
                "printParameters"
                , printParameters_function_type(&::IParameterized::printParameters)
                , default_printParameters_function_type(&Particle_wrapper::default_printParameters) );
        
        }
        { //::ISample::printSampleTree
        
            typedef void ( ::ISample::*printSampleTree_function_type)(  ) ;
            typedef void ( Particle_wrapper::*default_printSampleTree_function_type)(  ) ;
            
            Particle_exposer.def( 
                "printSampleTree"
                , printSampleTree_function_type(&::ISample::printSampleTree)
                , default_printSampleTree_function_type(&Particle_wrapper::default_printSampleTree) );
        
        }
        { //::IParameterized::registerParameter
        
            typedef void ( *default_registerParameter_function_type )( ::IParameterized &,::std::string const &,long unsigned int,::AttLimits const & );
            
            Particle_exposer.def( 
                "registerParameter"
                , default_registerParameter_function_type( &Particle_wrapper::default_registerParameter )
                , ( bp::arg("inst"), bp::arg("name"), bp::arg("parpointer"), bp::arg("limits")=AttLimits::limitless( ) )
                , "main method to register data address in the pool." );
        
        }
        { //::IParameterized::setParameterValue
        
            typedef bool ( ::IParameterized::*setParameterValue_function_type)( ::std::string const &,double ) ;
            typedef bool ( Particle_wrapper::*default_setParameterValue_function_type)( ::std::string const &,double ) ;
            
            Particle_exposer.def( 
                "setParameterValue"
                , setParameterValue_function_type(&::IParameterized::setParameterValue)
                , default_setParameterValue_function_type(&Particle_wrapper::default_setParameterValue)
                , ( bp::arg("name"), bp::arg("value") ) );
        
        }
        { //::IParameterized::setParametersAreChanged
        
            typedef void ( ::IParameterized::*setParametersAreChanged_function_type)(  ) ;
            typedef void ( Particle_wrapper::*default_setParametersAreChanged_function_type)(  ) ;
            
            Particle_exposer.def( 
                "setParametersAreChanged"
                , setParametersAreChanged_function_type(&::IParameterized::setParametersAreChanged)
                , default_setParametersAreChanged_function_type(&Particle_wrapper::default_setParametersAreChanged) );
        
        }
        { //::ICompositeSample::size
        
            typedef ::std::size_t ( ::ICompositeSample::*size_function_type)(  ) const;
            typedef ::std::size_t ( Particle_wrapper::*default_size_function_type)(  ) const;
            
            Particle_exposer.def( 
                "size"
                , size_function_type(&::ICompositeSample::size)
                , default_size_function_type(&Particle_wrapper::default_size) );
        
        }
        { //::ICloneable::transferToCPP
        
            typedef void ( ::ICloneable::*transferToCPP_function_type)(  ) ;
            typedef void ( Particle_wrapper::*default_transferToCPP_function_type)(  ) ;
            
            Particle_exposer.def( 
                "transferToCPP"
                , transferToCPP_function_type(&::ICloneable::transferToCPP)
                , default_transferToCPP_function_type(&Particle_wrapper::default_transferToCPP) );
        
        }
    }

}
