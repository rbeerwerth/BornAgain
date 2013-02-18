#include "TestFittingModule3.h"
#include "AttLimits.h"
#include "DrawHelper.h"
#include "Exceptions.h"
#include "ExperimentConstants.h"
#include "FitSuite.h"
#include "FitSuiteObserverFactory.h"
#include "FormFactors.h"
#include "Simulation.h"
#include "InterferenceFunction1DParaCrystal.h"
#include "InterferenceFunctionNone.h"
#include "IsGISAXSTools.h"
#include "LayerDecorator.h"
#include "MaterialManager.h"
#include "MathFunctions.h"
#include "MinimizerFactory.h"
#include "MultiLayer.h"
#include "OutputDataFunctions.h"
#include "Particle.h"
#include "ParticleDecoration.h"
#include "ResolutionFunction2DSimple.h"
#include "Units.h"

#include "TROOT.h"
#include "TLine.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TH2D.h"
#include "TPaveText.h"



TestFittingModule3::TestFittingModule3()
    : m_simulation(0)
    , m_sample(0)
    , m_fitSuite(0)
    , m_real_data(0)
{
    m_fitSuite = new FitSuite();

}


TestFittingModule3::~TestFittingModule3()
{
    delete m_simulation;
    delete m_sample;
    delete m_real_data;
    delete m_fitSuite;
}


void TestFittingModule3::execute()
{
    // initializing simulation, sample and data
    initializeSample();
    initializeSimulation();
    initializeRealData();

    // setting up fitSuite
    m_fitSuite->addFitParameter("*FormFactorCylinder/height", 4.0*Units::nanometer, 0.04*Units::nanometer, AttLimits::lowerLimited(0.01) );
    m_fitSuite->addFitParameter("*FormFactorCylinder/radius", 4.0*Units::nanometer, 0.04*Units::nanometer, AttLimits::lowerLimited(0.01) );
    m_fitSuite->addFitParameter("*FormFactorPrism3/half_side", 4.0*Units::nanometer, 0.04*Units::nanometer, AttLimits::lowerLimited(0.01) );
    m_fitSuite->addFitParameter("*FormFactorPrism3/height", 4.0*Units::nanometer, 0.04*Units::nanometer, AttLimits::lowerLimited(0.01) );

    // putting scans
    for(DataScan_t::iterator it=m_data_scans.begin(); it!= m_data_scans.end(); ++it) {
        m_fitSuite->addSimulationAndRealData(*m_simulation, *(*it));
    }

    m_fitSuite->setMinimizer( MinimizerFactory::createMinimizer("Minuit2", "Migrad") );
    m_fitSuite->attachObserver( FitSuiteObserverFactory::createPrintObserver() );
    m_fitSuite->attachObserver( FitSuiteObserverFactory::createDrawObserver() );

    m_fitSuite->runFit();
}


/* ************************************************************************* */
// initializing simulation
/* ************************************************************************* */
void TestFittingModule3::initializeSimulation()
{
    if( !m_sample ) {
        throw NullPointerException("TestFittingModule3::initializeSimulation() -> No sample defined");
    }
    delete m_simulation;
    m_simulation = new Simulation(mp_options);
    m_simulation->setSample(*m_sample);
    m_simulation->setDetectorParameters(100, 0.0*Units::degree, 2.0*Units::degree,100 , 0.0*Units::degree, 2.0*Units::degree);
    m_simulation->setBeamParameters(1.0*Units::angstrom, -0.2*Units::degree, 0.0*Units::degree);
    m_simulation->setBeamIntensity(1e10);

}


/* ************************************************************************* */
// initialize sample
/* ************************************************************************* */
void TestFittingModule3::initializeSample()
{
    delete m_sample;
    double cylinder_height = 5.0*Units::nanometer;
    double cylinder_radius = 5.0*Units::nanometer;
    double prism3_half_side = 5.0*Units::nanometer;
    double prism3_height = 5.0*Units::nanometer;
    MultiLayer *p_multi_layer = new MultiLayer();
    complex_t n_air(1.0, 0.0);
    complex_t n_substrate(1.0-6e-6, 2e-8);
    complex_t n_particle(1.0-6e-4, 2e-8);
    const IMaterial *p_air_material = MaterialManager::instance().addHomogeneousMaterial("Air", n_air);
    const IMaterial *p_substrate_material = MaterialManager::instance().addHomogeneousMaterial("Substrate", n_substrate);
    Layer air_layer;
    air_layer.setMaterial(p_air_material);
    Layer substrate_layer;
    substrate_layer.setMaterial(p_substrate_material);
    ParticleDecoration particle_decoration;
    particle_decoration.addParticle(new Particle(n_particle, new FormFactorCylinder(cylinder_height, cylinder_radius)),0.0, 0.2);
    particle_decoration.addParticle(new Particle(n_particle, new FormFactorPrism3(prism3_height, prism3_half_side)), 0.0, 0.8);
    particle_decoration.addInterferenceFunction(new InterferenceFunctionNone());
    LayerDecorator air_layer_decorator(air_layer, particle_decoration);
    p_multi_layer->addLayer(air_layer_decorator);
    p_multi_layer->addLayer(substrate_layer);
    m_sample = p_multi_layer;
}


/* ************************************************************************* */
// initializing real data
/* ************************************************************************* */
void TestFittingModule3::initializeRealData()
{
    if( !m_simulation || !m_sample ) throw NullPointerException("TestFittingModule3::initializeRealData() -> Error! No simulation of sample defined ");

    // generating 2D "real" data
    m_simulation->runSimulation();
    delete m_real_data;
    m_real_data = IsGISAXSTools::createNoisyData(*m_simulation->getOutputData());

    // setting up 1d scans by making slices on real data
    m_data_scans.clear();
    m_data_scans.push_back( OutputDataFunctions::selectRangeOnOneAxis(*m_real_data, NDetector2d::ALPHA_AXIS_NAME, 0.012, 0.012) );
    m_data_scans.push_back( OutputDataFunctions::selectRangeOnOneAxis(*m_real_data, NDetector2d::PHI_AXIS_NAME, 0.011, 0.011) );

    // drawing data and scans
    TCanvas *c1 = new TCanvas("c1","c1",1024, 768);
    c1->Divide(2,2);
    c1->cd(1); gPad->SetLogz();
    TH2D *hist1 = dynamic_cast<TH2D *>(IsGISAXSTools::getOutputDataTH123D( *m_real_data, "real_data"));
    hist1->Draw("COLZ");
    for(DataScan_t::iterator it=m_data_scans.begin(); it!= m_data_scans.end(); ++it) {
        TLine *line = IsGISAXSTools::getOutputDataScanLine(*(*it));
        line->DrawClone();
        delete line;
    }

    int npad(2);
    for(DataScan_t::iterator it=m_data_scans.begin(); it!= m_data_scans.end(); ++it, ++npad) {
        c1->cd(npad);
        TH1D *hist = IsGISAXSTools::getOutputDataScanHist(*(*it));
        hist->DrawCopy();
        delete hist;
    }

    c1->Update();
}


