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
#include "IFormFactorDecorator.pypp.h"

namespace bp = boost::python;

struct IFormFactorDecorator_wrapper : IFormFactorDecorator, bp::wrapper< IFormFactorDecorator > {

    IFormFactorDecorator_wrapper(::IFormFactor const & form_factor )
    : IFormFactorDecorator( boost::ref(form_factor) )
      , bp::wrapper< IFormFactorDecorator >(){
        // constructor
    m_pyobj = 0;
    }

    virtual void accept( ::ISampleVisitor * visitor ) const {
        bp::override func_accept = this->get_override( "accept" );
        func_accept( boost::python::ptr(visitor) );
    }

    virtual ::IFormFactorDecorator * clone(  ) const {
        bp::override func_clone = this->get_override( "clone" );
        return func_clone(  );
    }

    virtual double getHeight(  ) const  {
        if( bp::override func_getHeight = this->get_override( "getHeight" ) )
            return func_getHeight(  );
        else{
            return this->IFormFactorDecorator::getHeight(  );
        }
    }
    
    double default_getHeight(  ) const  {
        return IFormFactorDecorator::getHeight( );
    }

    virtual double getRadius(  ) const  {
        if( bp::override func_getRadius = this->get_override( "getRadius" ) )
            return func_getRadius(  );
        else{
            return this->IFormFactorDecorator::getRadius(  );
        }
    }
    
    double default_getRadius(  ) const  {
        return IFormFactorDecorator::getRadius( );
    }

    virtual double getVolume(  ) const  {
        if( bp::override func_getVolume = this->get_override( "getVolume" ) )
            return func_getVolume(  );
        else{
            return this->IFormFactorDecorator::getVolume(  );
        }
    }
    
    double default_getVolume(  ) const  {
        return IFormFactorDecorator::getVolume( );
    }

    virtual void setAmbientMaterial( ::IMaterial const & material ) {
        if( bp::override func_setAmbientMaterial = this->get_override( "setAmbientMaterial" ) )
            func_setAmbientMaterial( boost::ref(material) );
        else{
            this->IFormFactorDecorator::setAmbientMaterial( boost::ref(material) );
        }
    }
    
    void default_setAmbientMaterial( ::IMaterial const & material ) {
        IFormFactorDecorator::setAmbientMaterial( boost::ref(material) );
    }

    virtual ::ISample * cloneInvertB(  ) const  {
        if( bp::override func_cloneInvertB = this->get_override( "cloneInvertB" ) )
            return func_cloneInvertB(  );
        else{
            return this->ISample::cloneInvertB(  );
        }
    }
    
    ::ISample * default_cloneInvertB(  ) const  {
        return ISample::cloneInvertB( );
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

    virtual ::complex_t evaluate( ::WavevectorInfo const & wavevectors ) const {
        bp::override func_evaluate = this->get_override( "evaluate" );
        return func_evaluate( boost::ref(wavevectors) );
    }

    virtual ::ICompositeSample * getCompositeSample(  ) {
        if( bp::override func_getCompositeSample = this->get_override( "getCompositeSample" ) )
            return func_getCompositeSample(  );
        else{
            return this->ISample::getCompositeSample(  );
        }
    }
    
    ::ICompositeSample * default_getCompositeSample(  ) {
        return ISample::getCompositeSample( );
    }

    virtual ::ICompositeSample const * getCompositeSample(  ) const  {
        if( bp::override func_getCompositeSample = this->get_override( "getCompositeSample" ) )
            return func_getCompositeSample(  );
        else{
            return this->ISample::getCompositeSample(  );
        }
    }
    
    ::ICompositeSample const * default_getCompositeSample(  ) const  {
        return ISample::getCompositeSample( );
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

void register_IFormFactorDecorator_class(){

    { //::IFormFactorDecorator
        typedef bp::class_< IFormFactorDecorator_wrapper, bp::bases< IFormFactor >, std::auto_ptr< IFormFactorDecorator_wrapper >, boost::noncopyable > IFormFactorDecorator_exposer_t;
        IFormFactorDecorator_exposer_t IFormFactorDecorator_exposer = IFormFactorDecorator_exposer_t( "IFormFactorDecorator", "Encapsulates another formfactor and adds extra functionality (a scalar factor, a Debye-Waller factor,.", bp::init< IFormFactor const & >(( bp::arg("form_factor") )) );
        bp::scope IFormFactorDecorator_scope( IFormFactorDecorator_exposer );
        { //::IFormFactorDecorator::accept
        
            typedef void ( ::IFormFactorDecorator::*accept_function_type)( ::ISampleVisitor * ) const;
            
            IFormFactorDecorator_exposer.def( 
                "accept"
                , bp::pure_virtual( accept_function_type(&::IFormFactorDecorator::accept) )
                , ( bp::arg("visitor") ) );
        
        }
        { //::IFormFactorDecorator::clone
        
            typedef ::IFormFactorDecorator * ( ::IFormFactorDecorator::*clone_function_type)(  ) const;
            
            IFormFactorDecorator_exposer.def( 
                "clone"
                , bp::pure_virtual( clone_function_type(&::IFormFactorDecorator::clone) )
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::IFormFactorDecorator::getHeight
        
            typedef double ( ::IFormFactorDecorator::*getHeight_function_type)(  ) const;
            typedef double ( IFormFactorDecorator_wrapper::*default_getHeight_function_type)(  ) const;
            
            IFormFactorDecorator_exposer.def( 
                "getHeight"
                , getHeight_function_type(&::IFormFactorDecorator::getHeight)
                , default_getHeight_function_type(&IFormFactorDecorator_wrapper::default_getHeight) );
        
        }
        { //::IFormFactorDecorator::getRadius
        
            typedef double ( ::IFormFactorDecorator::*getRadius_function_type)(  ) const;
            typedef double ( IFormFactorDecorator_wrapper::*default_getRadius_function_type)(  ) const;
            
            IFormFactorDecorator_exposer.def( 
                "getRadius"
                , getRadius_function_type(&::IFormFactorDecorator::getRadius)
                , default_getRadius_function_type(&IFormFactorDecorator_wrapper::default_getRadius) );
        
        }
        { //::IFormFactorDecorator::getVolume
        
            typedef double ( ::IFormFactorDecorator::*getVolume_function_type)(  ) const;
            typedef double ( IFormFactorDecorator_wrapper::*default_getVolume_function_type)(  ) const;
            
            IFormFactorDecorator_exposer.def( 
                "getVolume"
                , getVolume_function_type(&::IFormFactorDecorator::getVolume)
                , default_getVolume_function_type(&IFormFactorDecorator_wrapper::default_getVolume) );
        
        }
        { //::IFormFactorDecorator::setAmbientMaterial
        
            typedef void ( ::IFormFactorDecorator::*setAmbientMaterial_function_type)( ::IMaterial const & ) ;
            typedef void ( IFormFactorDecorator_wrapper::*default_setAmbientMaterial_function_type)( ::IMaterial const & ) ;
            
            IFormFactorDecorator_exposer.def( 
                "setAmbientMaterial"
                , setAmbientMaterial_function_type(&::IFormFactorDecorator::setAmbientMaterial)
                , default_setAmbientMaterial_function_type(&IFormFactorDecorator_wrapper::default_setAmbientMaterial)
                , ( bp::arg("material") ) );
        
        }
        { //::ISample::cloneInvertB
        
            typedef ::ISample * ( ::ISample::*cloneInvertB_function_type)(  ) const;
            typedef ::ISample * ( IFormFactorDecorator_wrapper::*default_cloneInvertB_function_type)(  ) const;
            
            IFormFactorDecorator_exposer.def( 
                "cloneInvertB"
                , cloneInvertB_function_type(&::ISample::cloneInvertB)
                , default_cloneInvertB_function_type(&IFormFactorDecorator_wrapper::default_cloneInvertB)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ISample::containsMagneticMaterial
        
            typedef bool ( ::ISample::*containsMagneticMaterial_function_type)(  ) const;
            typedef bool ( IFormFactorDecorator_wrapper::*default_containsMagneticMaterial_function_type)(  ) const;
            
            IFormFactorDecorator_exposer.def( 
                "containsMagneticMaterial"
                , containsMagneticMaterial_function_type(&::ISample::containsMagneticMaterial)
                , default_containsMagneticMaterial_function_type(&IFormFactorDecorator_wrapper::default_containsMagneticMaterial) );
        
        }
        { //::IFormFactor::evaluate
        
            typedef ::complex_t ( ::IFormFactor::*evaluate_function_type)( ::WavevectorInfo const & ) const;
            
            IFormFactorDecorator_exposer.def( 
                "evaluate"
                , bp::pure_virtual( evaluate_function_type(&::IFormFactor::evaluate) )
                , ( bp::arg("wavevectors") )
                , "Returns scattering amplitude for complex wavevector bin @param k_i   incoming wavevector @param k_f_bin   outgoing wavevector bin \n\n:Parameters:\n  - 'k_i' - incoming wavevector\n  - 'k_f_bin' - outgoing wavevector bin\n" );
        
        }
        { //::ISample::getCompositeSample
        
            typedef ::ICompositeSample * ( ::ISample::*getCompositeSample_function_type)(  ) ;
            typedef ::ICompositeSample * ( IFormFactorDecorator_wrapper::*default_getCompositeSample_function_type)(  ) ;
            
            IFormFactorDecorator_exposer.def( 
                "getCompositeSample"
                , getCompositeSample_function_type(&::ISample::getCompositeSample)
                , default_getCompositeSample_function_type(&IFormFactorDecorator_wrapper::default_getCompositeSample)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ISample::getCompositeSample
        
            typedef ::ICompositeSample const * ( ::ISample::*getCompositeSample_function_type)(  ) const;
            typedef ::ICompositeSample const * ( IFormFactorDecorator_wrapper::*default_getCompositeSample_function_type)(  ) const;
            
            IFormFactorDecorator_exposer.def( 
                "getCompositeSample"
                , getCompositeSample_function_type(&::ISample::getCompositeSample)
                , default_getCompositeSample_function_type(&IFormFactorDecorator_wrapper::default_getCompositeSample)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ISample::printSampleTree
        
            typedef void ( ::ISample::*printSampleTree_function_type)(  ) ;
            typedef void ( IFormFactorDecorator_wrapper::*default_printSampleTree_function_type)(  ) ;
            
            IFormFactorDecorator_exposer.def( 
                "printSampleTree"
                , printSampleTree_function_type(&::ISample::printSampleTree)
                , default_printSampleTree_function_type(&IFormFactorDecorator_wrapper::default_printSampleTree) );
        
        }
        { //::ICloneable::transferToCPP
        
            typedef void ( ::ICloneable::*transferToCPP_function_type)(  ) ;
            typedef void ( IFormFactorDecorator_wrapper::*default_transferToCPP_function_type)(  ) ;
            
            IFormFactorDecorator_exposer.def( 
                "transferToCPP"
                , transferToCPP_function_type(&::ICloneable::transferToCPP)
                , default_transferToCPP_function_type(&IFormFactorDecorator_wrapper::default_transferToCPP) );
        
        }
    }

}
