#include "IsGISAXS15.h"
#include "MultiLayer.h"
#include "ParticleDecoration.h"
#include "LayerDecorator.h"
#include "ParticleBuilder.h"
#include "InterferenceFunction1DParaCrystal.h"
#include "FormFactorCylinder.h"
#include "Simulation.h"
#include "Units.h"
#include "MaterialManager.h"
#include "StochasticGaussian.h"
#include "StochasticSampledParameter.h"
#include "OutputDataIOFactory.h"
#include "Utils.h"

#include <iostream>
#include <cmath>


FunctionalTests::IsGISAXS15::IsGISAXS15()
    : m_name("IsGISAXS15")
    , m_description("Size spacing correlation approximation")
    , m_result(0)
{ }


void FunctionalTests::IsGISAXS15::run()
{
    // building sample
    MultiLayer multi_layer;
    complex_t n_particle(1.0-6e-4, 2e-8);
    const IMaterial *p_air_material = MaterialManager::instance().addHomogeneousMaterial("Air", 1.0, 0.0);
    Layer air_layer;
    air_layer.setMaterial(p_air_material);
    InterferenceFunction1DParaCrystal *p_interference_function = new InterferenceFunction1DParaCrystal(15.0*Units::nanometer,5*Units::nanometer, 1e3*Units::nanometer);
    p_interference_function->setKappa(4.02698);
    ParticleDecoration particle_decoration;
    Particle particle_prototype(n_particle, new FormFactorCylinder(5.0*Units::nanometer, 5.0*Units::nanometer));
    StochasticSampledParameter stochastic_radius(StochasticDoubleGaussian(5.0*Units::nanometer, 1.25*Units::nanometer),30, 2);
    ParticleBuilder particle_builder;
    particle_builder.setPrototype(particle_prototype, "/Particle/FormFactorCylinder/radius", stochastic_radius);
    particle_builder.plantParticles(particle_decoration);

    // Set height of each particle to its radius (H/R fixed)
    ParameterPool *p_parameters = particle_decoration.createParameterTree();
    //int nbr_replacements = p_parameters->fixRatioBetweenParameters("height", "radius", 1.0);
    //std::cout << "Number of replacements: " << nbr_replacements << std::endl;
    p_parameters->fixRatioBetweenParameters("height", "radius", 1.0);

    particle_decoration.addInterferenceFunction(p_interference_function);
    LayerDecorator air_layer_decorator(air_layer, particle_decoration);
    multi_layer.addLayer(air_layer_decorator);

   // building simulation
    Simulation simulation;
    simulation.setDetectorParameters(150, 0.05*Units::degree, 1.5*Units::degree, 150, 0.05*Units::degree, 1.5*Units::degree, true);
    simulation.setBeamParameters(1.0*Units::angstrom, -0.2*Units::degree, 0.0*Units::degree);
    SimulationParameters sim_params;
    sim_params.me_if_approx = SimulationParameters::SSCA;
    simulation.setSimulationParameters(sim_params);
    simulation.setSample(multi_layer);

    // running simulation and copying the data
    simulation.runSimulation();
    m_result = simulation.getOutputDataClone();
}


int FunctionalTests::IsGISAXS15::analyseResults()
{
    const double threshold(1e-10);

    // retrieving reference data
    std::string filename = Utils::FileSystem::GetHomePath() + "/Tests/FunctionalTests/TestCore/IsGISAXS15/isgisaxs15_reference.ima.gz";
    OutputData<double > *reference = OutputDataIOFactory::getOutputData(filename);

    // calculating average relative difference
    *m_result -= *reference;
    *m_result /= *reference;

    double diff(0);
    for(OutputData<double>::const_iterator it=m_result->begin(); it!=m_result->end(); ++it) {
        diff+= std::fabs(*it);
    }
    diff /= m_result->getAllocatedSize();

    bool status_ok(true);
    if( diff > threshold || std::isnan(diff)) status_ok=false;

    std::cout << m_name << " " << m_description << " " << (status_ok ? "[OK]" : "[FAILED]") << std::endl;
    return (int)status_ok;
}


#ifdef STANDALONE
int main()
{
    FunctionalTests::IsGISAXS15 test;
    test.run();

    return test.analyseResults();
}
#endif
