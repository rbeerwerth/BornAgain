// This file has been generated by Py++.

#include "Macros.h"
GCC_DIAG_OFF(unused-parameter);
GCC_DIAG_OFF(missing-field-initializers);
#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp"
GCC_DIAG_ON(unused-parameter);
GCC_DIAG_ON(missing-field-initializers);
#include "BasicVector3D.h"
#include "Bin.h"
#include "Crystal.h"
#include "DiffuseParticleInfo.h"
#include "FTDistributions.h"
#include "FormFactorBox.h"
#include "FormFactorCrystal.h"
#include "FormFactorCylinder.h"
#include "FormFactorDecoratorDebyeWaller.h"
#include "FormFactorFullSphere.h"
#include "FormFactorGauss.h"
#include "FormFactorLorentz.h"
#include "FormFactorParallelepiped.h"
#include "FormFactorPrism3.h"
#include "FormFactorPyramid.h"
#include "FormFactorSphereGaussianRadius.h"
#include "HomogeneousMaterial.h"
#include "ICloneable.h"
#include "IClusteredParticles.h"
#include "ICompositeSample.h"
#include "IDecoration.h"
#include "IFormFactor.h"
#include "IFormFactorBorn.h"
#include "IFormFactorDecorator.h"
#include "IInterferenceFunction.h"
#include "IMaterial.h"
#include "IParameterized.h"
#include "ISample.h"
#include "ISampleBuilder.h"
#include "ISelectionRule.h"
#include "ISingleton.h"
#include "Instrument.h"
#include "InterferenceFunction1DParaCrystal.h"
#include "InterferenceFunction2DLattice.h"
#include "InterferenceFunction2DParaCrystal.h"
#include "InterferenceFunctionNone.h"
#include "Lattice.h"
#include "Lattice2DIFParameters.h"
#include "LatticeBasis.h"
#include "Layer.h"
#include "LayerDecorator.h"
#include "LayerRoughness.h"
#include "Lattice2DIFParameters.h"
#include "MaterialManager.h"
#include "MesoCrystal.h"
#include "MultiLayer.h"
#include "OpticalFresnel.h"
#include "ParameterPool.h"
#include "Particle.h"
#include "ParticleBuilder.h"
#include "ParticleCoreShell.h"
#include "ParticleDecoration.h"
#include "ParticleInfo.h"
#include "PositionParticleInfo.h"
#include "PythonOutputData.h"
#include "PythonPlusplusHelper.h"
#include "RealParameterWrapper.h"
#include "Simulation.h"
#include "SimulationParameters.h"
#include "IStochasticParameter.h"
#include "StochasticGaussian.h"
#include "StochasticSampledParameter.h"
#include "StochasticDoubleGate.h"
#include "Transform3D.h"
#include "Types.h"
#include "Units.h"
#include "cvector_t.pypp.h"

namespace bp = boost::python;

void register_cvector_t_class(){

    { //::Geometry::BasicVector3D< std::complex< double > >
        typedef bp::class_< Geometry::BasicVector3D< std::complex< double > > > cvector_t_exposer_t;
        cvector_t_exposer_t cvector_t_exposer = cvector_t_exposer_t( "cvector_t", bp::init< >() );
        bp::scope cvector_t_scope( cvector_t_exposer );
        bp::scope().attr("X") = (int)Geometry::BasicVector3D<std::complex<double> >::X;
        bp::scope().attr("Y") = (int)Geometry::BasicVector3D<std::complex<double> >::Y;
        bp::scope().attr("Z") = (int)Geometry::BasicVector3D<std::complex<double> >::Z;
        bp::scope().attr("NUM_COORDINATES") = (int)Geometry::BasicVector3D<std::complex<double> >::NUM_COORDINATES;
        bp::scope().attr("SIZE") = (int)Geometry::BasicVector3D<std::complex<double> >::SIZE;
        cvector_t_exposer.def( bp::init< std::complex< double > const &, std::complex< double > const &, std::complex< double > const & >(( bp::arg("x1"), bp::arg("y1"), bp::arg("z1") )) );
        { //::Geometry::BasicVector3D< std::complex< double > >::cross
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::Geometry::BasicVector3D< std::complex< double > > ( exported_class_t::*cross_function_type )( ::Geometry::BasicVector3D< std::complex< double > > const & ) const;
            
            cvector_t_exposer.def( 
                "cross"
                , cross_function_type( &::Geometry::BasicVector3D< std::complex< double > >::cross )
                , ( bp::arg("v") ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::dot
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*dot_function_type )( ::Geometry::BasicVector3D< std::complex< double > > const & ) const;
            
            cvector_t_exposer.def( 
                "dot"
                , dot_function_type( &::Geometry::BasicVector3D< std::complex< double > >::dot )
                , ( bp::arg("v") ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::mag
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*mag_function_type )(  ) const;
            
            cvector_t_exposer.def( 
                "mag"
                , mag_function_type( &::Geometry::BasicVector3D< std::complex< double > >::mag ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::mag2
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*mag2_function_type )(  ) const;
            
            cvector_t_exposer.def( 
                "mag2"
                , mag2_function_type( &::Geometry::BasicVector3D< std::complex< double > >::mag2 ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::magxy
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*magxy_function_type )(  ) const;
            
            cvector_t_exposer.def( 
                "magxy"
                , magxy_function_type( &::Geometry::BasicVector3D< std::complex< double > >::magxy ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::magxy2
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*magxy2_function_type )(  ) const;
            
            cvector_t_exposer.def( 
                "magxy2"
                , magxy2_function_type( &::Geometry::BasicVector3D< std::complex< double > >::magxy2 ) );
        
        }
        cvector_t_exposer.def( bp::self *= bp::other< double >() );
        cvector_t_exposer.def( bp::self += bp::self );
        cvector_t_exposer.def( bp::self -= bp::self );
        cvector_t_exposer.def( bp::self /= bp::other< double >() );
        { //::Geometry::BasicVector3D< std::complex< double > >::operator=
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::Geometry::BasicVector3D< std::complex< double > > & ( exported_class_t::*assign_function_type )( ::Geometry::BasicVector3D< std::complex< double > > const & ) ;
            
            cvector_t_exposer.def( 
                "assign"
                , assign_function_type( &::Geometry::BasicVector3D< std::complex< double > >::operator= )
                , ( bp::arg("v") )
                , bp::return_self< >() );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::operator[]
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*__getitem___function_type )( int ) const;
            
            cvector_t_exposer.def( 
                "__getitem__"
                , __getitem___function_type( &::Geometry::BasicVector3D< std::complex< double > >::operator[] )
                , ( bp::arg("i") ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::operator[]
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > & ( exported_class_t::*__getitem___function_type )( int ) ;
            
            cvector_t_exposer.def( 
                "__getitem__"
                , __getitem___function_type( &::Geometry::BasicVector3D< std::complex< double > >::operator[] )
                , ( bp::arg("i") )
                , bp::return_internal_reference< >() );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::setLambdaAlphaPhi
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef void ( exported_class_t::*setLambdaAlphaPhi_function_type )( ::std::complex< double > const &,::std::complex< double > const &,::std::complex< double > const & ) ;
            
            cvector_t_exposer.def( 
                "setLambdaAlphaPhi"
                , setLambdaAlphaPhi_function_type( &::Geometry::BasicVector3D< std::complex< double > >::setLambdaAlphaPhi )
                , ( bp::arg("_lambda"), bp::arg("_alpha"), bp::arg("_phi") ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::setX
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef void ( exported_class_t::*setX_function_type )( ::std::complex< double > const & ) ;
            
            cvector_t_exposer.def( 
                "setX"
                , setX_function_type( &::Geometry::BasicVector3D< std::complex< double > >::setX )
                , ( bp::arg("a") ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::setXYZ
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef void ( exported_class_t::*setXYZ_function_type )( ::std::complex< double > const &,::std::complex< double > const &,::std::complex< double > const & ) ;
            
            cvector_t_exposer.def( 
                "setXYZ"
                , setXYZ_function_type( &::Geometry::BasicVector3D< std::complex< double > >::setXYZ )
                , ( bp::arg("x1"), bp::arg("y1"), bp::arg("z1") ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::setY
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef void ( exported_class_t::*setY_function_type )( ::std::complex< double > const & ) ;
            
            cvector_t_exposer.def( 
                "setY"
                , setY_function_type( &::Geometry::BasicVector3D< std::complex< double > >::setY )
                , ( bp::arg("a") ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::setZ
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef void ( exported_class_t::*setZ_function_type )( ::std::complex< double > const & ) ;
            
            cvector_t_exposer.def( 
                "setZ"
                , setZ_function_type( &::Geometry::BasicVector3D< std::complex< double > >::setZ )
                , ( bp::arg("a") ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::transform
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::Geometry::BasicVector3D< std::complex< double > > & ( exported_class_t::*transform_function_type )( ::Geometry::Transform3D const & ) ;
            
            cvector_t_exposer.def( 
                "transform"
                , transform_function_type( &::Geometry::BasicVector3D< std::complex< double > >::transform )
                , ( bp::arg("m") )
                , bp::return_value_policy< bp::reference_existing_object >() );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::x
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*x_function_type )(  ) const;
            
            cvector_t_exposer.def( 
                "x"
                , x_function_type( &::Geometry::BasicVector3D< std::complex< double > >::x ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::y
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*y_function_type )(  ) const;
            
            cvector_t_exposer.def( 
                "y"
                , y_function_type( &::Geometry::BasicVector3D< std::complex< double > >::y ) );
        
        }
        { //::Geometry::BasicVector3D< std::complex< double > >::z
        
            typedef Geometry::BasicVector3D< std::complex< double > > exported_class_t;
            typedef ::std::complex< double > ( exported_class_t::*z_function_type )(  ) const;
            
            cvector_t_exposer.def( 
                "z"
                , z_function_type( &::Geometry::BasicVector3D< std::complex< double > >::z ) );
        
        }
        cvector_t_exposer.def( bp::self != bp::self );
        cvector_t_exposer.def( bp::self + bp::self );
        cvector_t_exposer.def( +bp::self );
        cvector_t_exposer.def( bp::self - bp::self );
        cvector_t_exposer.def( -bp::self );
        cvector_t_exposer.def( bp::self / bp::other< std::complex< double > >() );
        cvector_t_exposer.def( bp::self_ns::str( bp::self ) );
        cvector_t_exposer.def( bp::self == bp::self );
    }

}
