// This file has been generated by Py++.

#include "Macros.h"
GCC_DIAG_OFF(unused-parameter);
GCC_DIAG_OFF(missing-field-initializers);
#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp"
GCC_DIAG_ON(unused-parameter);
GCC_DIAG_ON(missing-field-initializers);
#include "BasicVector3D.h"
#include "Experiment.h"
#include "FormFactorCrystal.h"
#include "FormFactorCylinder.h"
#include "FormFactorDecoratorDebyeWaller.h"
#include "FormFactorFullSphere.h"
#include "FormFactorGauss.h"
#include "FormFactorLorentz.h"
#include "FormFactorPrism3.h"
#include "FormFactorPyramid.h"
#include "FormFactorSphereGaussianRadius.h"
#include "GISASExperiment.h"
#include "HomogeneousMaterial.h"
#include "IClusteredParticles.h"
#include "ICompositeSample.h"
#include "IFormFactor.h"
#include "IFormFactorBorn.h"
#include "IFormFactorDecorator.h"
#include "IInterferenceFunction.h"
#include "InterferenceFunctionNone.h"
#include "InterferenceFunction1DParaCrystal.h"
#include "IMaterial.h"
#include "IParameterized.h"
#include "ISample.h"
#include "ISampleBuilder.h"
#include "ISelectionRule.h"
#include "ISingleton.h"
#include "Lattice.h"
#include "LatticeBasis.h"
#include "Layer.h"
#include "LayerDecorator.h"
#include "LayerRoughness.h"
#include "MaterialManager.h"
#include "MesoCrystal.h"
#include "MultiLayer.h"
#include "Particle.h"
#include "Crystal.h"
#include "ParticleDecoration.h"
#include "OpticalFresnel.h"
#include "ParameterPool.h"
#include "ParticleInfo.h"
#include "DiffuseParticleInfo.h"
#include "PythonOutputData.h"
#include "PythonPlusplusHelper.h"
#include "RealParameterWrapper.h"
#include "Transform3D.h"
#include "Units.h"
#include "Types.h"
#include "IMaterial.pypp.h"

namespace bp = boost::python;

void register_IMaterial_class(){

    { //::IMaterial
        typedef bp::class_< IMaterial > IMaterial_exposer_t;
        IMaterial_exposer_t IMaterial_exposer = IMaterial_exposer_t( "IMaterial", bp::init< >() );
        bp::scope IMaterial_scope( IMaterial_exposer );
        IMaterial_exposer.def( bp::init< std::string const & >(( bp::arg("name") )) );
        IMaterial_exposer.def( bp::init< IMaterial const & >(( bp::arg("other") )) );
        { //::IMaterial::operator=
        
            typedef ::IMaterial & ( ::IMaterial::*assign_function_type )( ::IMaterial const & ) ;
            
            IMaterial_exposer.def( 
                "assign"
                , assign_function_type( &::IMaterial::operator= )
                , ( bp::arg("other") )
                , bp::return_self< >() );
        
        }
        IMaterial_exposer.def( bp::self_ns::str( bp::self ) );
    }

}
