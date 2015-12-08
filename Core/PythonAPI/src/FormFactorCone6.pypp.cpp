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
#include "FormFactorCone6.pypp.h"

namespace bp = boost::python;

struct FormFactorCone6_wrapper : FormFactorCone6, bp::wrapper< FormFactorCone6 > {

    FormFactorCone6_wrapper(double radius, double height, double alpha )
    : FormFactorCone6( radius, height, alpha )
      , bp::wrapper< FormFactorCone6 >(){
        // constructor
    m_pyobj = 0;
    }

    virtual ::FormFactorCone6 * clone(  ) const  {
        if( bp::override func_clone = this->get_override( "clone" ) )
            return func_clone(  );
        else{
            return this->FormFactorCone6::clone(  );
        }
    }
    
    ::FormFactorCone6 * default_clone(  ) const  {
        return FormFactorCone6::clone( );
    }

    virtual ::complex_t evaluate_for_q( ::cvector_t const & q ) const  {
        if( bp::override func_evaluate_for_q = this->get_override( "evaluate_for_q" ) )
            return func_evaluate_for_q( boost::ref(q) );
        else{
            return this->FormFactorCone6::evaluate_for_q( boost::ref(q) );
        }
    }
    
    ::complex_t default_evaluate_for_q( ::cvector_t const & q ) const  {
        return FormFactorCone6::evaluate_for_q( boost::ref(q) );
    }

    virtual double getAlpha(  ) const  {
        if( bp::override func_getAlpha = this->get_override( "getAlpha" ) )
            return func_getAlpha(  );
        else{
            return this->FormFactorCone6::getAlpha(  );
        }
    }
    
    double default_getAlpha(  ) const  {
        return FormFactorCone6::getAlpha( );
    }

    virtual double getHeight(  ) const  {
        if( bp::override func_getHeight = this->get_override( "getHeight" ) )
            return func_getHeight(  );
        else{
            return this->FormFactorCone6::getHeight(  );
        }
    }
    
    double default_getHeight(  ) const  {
        return FormFactorCone6::getHeight( );
    }

    virtual double getRadius(  ) const  {
        if( bp::override func_getRadius = this->get_override( "getRadius" ) )
            return func_getRadius(  );
        else{
            return this->FormFactorCone6::getRadius(  );
        }
    }
    
    double default_getRadius(  ) const  {
        return FormFactorCone6::getRadius( );
    }

    virtual void setAlpha( double alpha ) {
        if( bp::override func_setAlpha = this->get_override( "setAlpha" ) )
            func_setAlpha( alpha );
        else{
            this->FormFactorCone6::setAlpha( alpha );
        }
    }
    
    void default_setAlpha( double alpha ) {
        FormFactorCone6::setAlpha( alpha );
    }

    virtual void setHeight( double height ) {
        if( bp::override func_setHeight = this->get_override( "setHeight" ) )
            func_setHeight( height );
        else{
            this->FormFactorCone6::setHeight( height );
        }
    }
    
    void default_setHeight( double height ) {
        FormFactorCone6::setHeight( height );
    }

    virtual void setRadius( double radius ) {
        if( bp::override func_setRadius = this->get_override( "setRadius" ) )
            func_setRadius( radius );
        else{
            this->FormFactorCone6::setRadius( radius );
        }
    }
    
    void default_setRadius( double radius ) {
        FormFactorCone6::setRadius( radius );
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

    virtual ::complex_t evaluate( ::WavevectorInfo const & wavevectors ) const  {
        if( bp::override func_evaluate = this->get_override( "evaluate" ) )
            return func_evaluate( boost::ref(wavevectors) );
        else{
            return this->IFormFactorBorn::evaluate( boost::ref(wavevectors) );
        }
    }
    
    ::complex_t default_evaluate( ::WavevectorInfo const & wavevectors ) const  {
        return IFormFactorBorn::evaluate( boost::ref(wavevectors) );
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

    virtual double getVolume(  ) const  {
        if( bp::override func_getVolume = this->get_override( "getVolume" ) )
            return func_getVolume(  );
        else{
            return this->IFormFactorBorn::getVolume(  );
        }
    }
    
    double default_getVolume(  ) const  {
        return IFormFactorBorn::getVolume( );
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

    virtual void setAmbientMaterial( ::IMaterial const & material ) {
        if( bp::override func_setAmbientMaterial = this->get_override( "setAmbientMaterial" ) )
            func_setAmbientMaterial( boost::ref(material) );
        else{
            this->IFormFactor::setAmbientMaterial( boost::ref(material) );
        }
    }
    
    void default_setAmbientMaterial( ::IMaterial const & material ) {
        IFormFactor::setAmbientMaterial( boost::ref(material) );
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

void register_FormFactorCone6_class(){

    { //::FormFactorCone6
        typedef bp::class_< FormFactorCone6_wrapper, bp::bases< IFormFactorBorn >, std::auto_ptr< FormFactorCone6_wrapper >, boost::noncopyable > FormFactorCone6_exposer_t;
        FormFactorCone6_exposer_t FormFactorCone6_exposer = FormFactorCone6_exposer_t( "FormFactorCone6", "The formfactor of a cone6.", bp::init< double, double, double >(( bp::arg("radius"), bp::arg("height"), bp::arg("alpha") ), "Cone6 constructor.\n\n:Parameters:\n  - 'radius' - of hexagonal base (different from R in IsGisaxs)\n  - 'height' - of Cone6\n  - 'angle' - in radians between base and facet\n") );
        bp::scope FormFactorCone6_scope( FormFactorCone6_exposer );
        { //::FormFactorCone6::clone
        
            typedef ::FormFactorCone6 * ( ::FormFactorCone6::*clone_function_type)(  ) const;
            typedef ::FormFactorCone6 * ( FormFactorCone6_wrapper::*default_clone_function_type)(  ) const;
            
            FormFactorCone6_exposer.def( 
                "clone"
                , clone_function_type(&::FormFactorCone6::clone)
                , default_clone_function_type(&FormFactorCone6_wrapper::default_clone)
                , bp::return_value_policy< bp::manage_new_object >() );
        
        }
        { //::FormFactorCone6::evaluate_for_q
        
            typedef ::complex_t ( ::FormFactorCone6::*evaluate_for_q_function_type)( ::cvector_t const & ) const;
            typedef ::complex_t ( FormFactorCone6_wrapper::*default_evaluate_for_q_function_type)( ::cvector_t const & ) const;
            
            FormFactorCone6_exposer.def( 
                "evaluate_for_q"
                , evaluate_for_q_function_type(&::FormFactorCone6::evaluate_for_q)
                , default_evaluate_for_q_function_type(&FormFactorCone6_wrapper::default_evaluate_for_q)
                , ( bp::arg("q") ) );
        
        }
        { //::FormFactorCone6::getAlpha
        
            typedef double ( ::FormFactorCone6::*getAlpha_function_type)(  ) const;
            typedef double ( FormFactorCone6_wrapper::*default_getAlpha_function_type)(  ) const;
            
            FormFactorCone6_exposer.def( 
                "getAlpha"
                , getAlpha_function_type(&::FormFactorCone6::getAlpha)
                , default_getAlpha_function_type(&FormFactorCone6_wrapper::default_getAlpha) );
        
        }
        { //::FormFactorCone6::getHeight
        
            typedef double ( ::FormFactorCone6::*getHeight_function_type)(  ) const;
            typedef double ( FormFactorCone6_wrapper::*default_getHeight_function_type)(  ) const;
            
            FormFactorCone6_exposer.def( 
                "getHeight"
                , getHeight_function_type(&::FormFactorCone6::getHeight)
                , default_getHeight_function_type(&FormFactorCone6_wrapper::default_getHeight) );
        
        }
        { //::FormFactorCone6::getRadius
        
            typedef double ( ::FormFactorCone6::*getRadius_function_type)(  ) const;
            typedef double ( FormFactorCone6_wrapper::*default_getRadius_function_type)(  ) const;
            
            FormFactorCone6_exposer.def( 
                "getRadius"
                , getRadius_function_type(&::FormFactorCone6::getRadius)
                , default_getRadius_function_type(&FormFactorCone6_wrapper::default_getRadius) );
        
        }
        { //::FormFactorCone6::setAlpha
        
            typedef void ( ::FormFactorCone6::*setAlpha_function_type)( double ) ;
            typedef void ( FormFactorCone6_wrapper::*default_setAlpha_function_type)( double ) ;
            
            FormFactorCone6_exposer.def( 
                "setAlpha"
                , setAlpha_function_type(&::FormFactorCone6::setAlpha)
                , default_setAlpha_function_type(&FormFactorCone6_wrapper::default_setAlpha)
                , ( bp::arg("alpha") ) );
        
        }
        { //::FormFactorCone6::setHeight
        
            typedef void ( ::FormFactorCone6::*setHeight_function_type)( double ) ;
            typedef void ( FormFactorCone6_wrapper::*default_setHeight_function_type)( double ) ;
            
            FormFactorCone6_exposer.def( 
                "setHeight"
                , setHeight_function_type(&::FormFactorCone6::setHeight)
                , default_setHeight_function_type(&FormFactorCone6_wrapper::default_setHeight)
                , ( bp::arg("height") ) );
        
        }
        { //::FormFactorCone6::setRadius
        
            typedef void ( ::FormFactorCone6::*setRadius_function_type)( double ) ;
            typedef void ( FormFactorCone6_wrapper::*default_setRadius_function_type)( double ) ;
            
            FormFactorCone6_exposer.def( 
                "setRadius"
                , setRadius_function_type(&::FormFactorCone6::setRadius)
                , default_setRadius_function_type(&FormFactorCone6_wrapper::default_setRadius)
                , ( bp::arg("radius") ) );
        
        }
        { //::ISample::cloneInvertB
        
            typedef ::ISample * ( ::ISample::*cloneInvertB_function_type)(  ) const;
            typedef ::ISample * ( FormFactorCone6_wrapper::*default_cloneInvertB_function_type)(  ) const;
            
            FormFactorCone6_exposer.def( 
                "cloneInvertB"
                , cloneInvertB_function_type(&::ISample::cloneInvertB)
                , default_cloneInvertB_function_type(&FormFactorCone6_wrapper::default_cloneInvertB)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ISample::containsMagneticMaterial
        
            typedef bool ( ::ISample::*containsMagneticMaterial_function_type)(  ) const;
            typedef bool ( FormFactorCone6_wrapper::*default_containsMagneticMaterial_function_type)(  ) const;
            
            FormFactorCone6_exposer.def( 
                "containsMagneticMaterial"
                , containsMagneticMaterial_function_type(&::ISample::containsMagneticMaterial)
                , default_containsMagneticMaterial_function_type(&FormFactorCone6_wrapper::default_containsMagneticMaterial) );
        
        }
        { //::IFormFactorBorn::evaluate
        
            typedef ::complex_t ( ::IFormFactorBorn::*evaluate_function_type)( ::WavevectorInfo const & ) const;
            typedef ::complex_t ( FormFactorCone6_wrapper::*default_evaluate_function_type)( ::WavevectorInfo const & ) const;
            
            FormFactorCone6_exposer.def( 
                "evaluate"
                , evaluate_function_type(&::IFormFactorBorn::evaluate)
                , default_evaluate_function_type(&FormFactorCone6_wrapper::default_evaluate)
                , ( bp::arg("wavevectors") ) );
        
        }
        { //::ISample::getCompositeSample
        
            typedef ::ICompositeSample * ( ::ISample::*getCompositeSample_function_type)(  ) ;
            typedef ::ICompositeSample * ( FormFactorCone6_wrapper::*default_getCompositeSample_function_type)(  ) ;
            
            FormFactorCone6_exposer.def( 
                "getCompositeSample"
                , getCompositeSample_function_type(&::ISample::getCompositeSample)
                , default_getCompositeSample_function_type(&FormFactorCone6_wrapper::default_getCompositeSample)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::ISample::getCompositeSample
        
            typedef ::ICompositeSample const * ( ::ISample::*getCompositeSample_function_type)(  ) const;
            typedef ::ICompositeSample const * ( FormFactorCone6_wrapper::*default_getCompositeSample_function_type)(  ) const;
            
            FormFactorCone6_exposer.def( 
                "getCompositeSample"
                , getCompositeSample_function_type(&::ISample::getCompositeSample)
                , default_getCompositeSample_function_type(&FormFactorCone6_wrapper::default_getCompositeSample)
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::IFormFactorBorn::getVolume
        
            typedef double ( ::IFormFactorBorn::*getVolume_function_type)(  ) const;
            typedef double ( FormFactorCone6_wrapper::*default_getVolume_function_type)(  ) const;
            
            FormFactorCone6_exposer.def( 
                "getVolume"
                , getVolume_function_type(&::IFormFactorBorn::getVolume)
                , default_getVolume_function_type(&FormFactorCone6_wrapper::default_getVolume) );
        
        }
        { //::ISample::printSampleTree
        
            typedef void ( ::ISample::*printSampleTree_function_type)(  ) ;
            typedef void ( FormFactorCone6_wrapper::*default_printSampleTree_function_type)(  ) ;
            
            FormFactorCone6_exposer.def( 
                "printSampleTree"
                , printSampleTree_function_type(&::ISample::printSampleTree)
                , default_printSampleTree_function_type(&FormFactorCone6_wrapper::default_printSampleTree) );
        
        }
        { //::IFormFactor::setAmbientMaterial
        
            typedef void ( ::IFormFactor::*setAmbientMaterial_function_type)( ::IMaterial const & ) ;
            typedef void ( FormFactorCone6_wrapper::*default_setAmbientMaterial_function_type)( ::IMaterial const & ) ;
            
            FormFactorCone6_exposer.def( 
                "setAmbientMaterial"
                , setAmbientMaterial_function_type(&::IFormFactor::setAmbientMaterial)
                , default_setAmbientMaterial_function_type(&FormFactorCone6_wrapper::default_setAmbientMaterial)
                , ( bp::arg("material") ) );
        
        }
        { //::ICloneable::transferToCPP
        
            typedef void ( ::ICloneable::*transferToCPP_function_type)(  ) ;
            typedef void ( FormFactorCone6_wrapper::*default_transferToCPP_function_type)(  ) ;
            
            FormFactorCone6_exposer.def( 
                "transferToCPP"
                , transferToCPP_function_type(&::ICloneable::transferToCPP)
                , default_transferToCPP_function_type(&FormFactorCone6_wrapper::default_transferToCPP) );
        
        }
    }

}
