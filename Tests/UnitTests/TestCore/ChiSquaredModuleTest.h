#ifndef CHISQUAREDMODULETEST_H
#define CHISQUAREDMODULETEST_H

#include "ChiSquaredModule.h"
#include "IFittingDataSelector.h"
#include "ISquaredFunction.h"
#include "ExperimentConstants.h"

#include "gtest/gtest.h"


class ChiSquaredModuleTest : public ::testing::Test
{
protected:
    ChiSquaredModuleTest();
    virtual ~ChiSquaredModuleTest();

    ChiSquaredModule m_chi_empty;
    ChiSquaredModule m_chi_default;
    OutputData<double > m_real_data;
    OutputData<double > m_simul_data;
};


ChiSquaredModuleTest::ChiSquaredModuleTest()
{
    m_real_data.addAxis(NDetector2d::PHI_AXIS_NAME, 10, 0.0, 10.0);
    m_real_data.addAxis(NDetector2d::ALPHA_AXIS_NAME, 10, 0.0, 10.0);
    m_real_data.setAllTo(1.0);
    m_simul_data.copyFrom(m_real_data);
    m_simul_data.setAllTo(1.1);

    // default module with set of real and simulated data
    m_chi_default.setRealAndSimulatedData(m_real_data, m_simul_data);

}


ChiSquaredModuleTest::~ChiSquaredModuleTest()
{

}


TEST_F(ChiSquaredModuleTest, InitialState)
{
    EXPECT_EQ( NULL, m_chi_empty.getRealData());
    EXPECT_EQ( NULL, m_chi_empty.getSimulationData());
    EXPECT_TRUE( dynamic_cast<const SquaredFunctionDefault *>(m_chi_empty.getSquaredFunction()));
    EXPECT_TRUE( dynamic_cast<const DefaultAllDataSelector *>(m_chi_empty.getFittingDataSelector()));
    EXPECT_EQ( NULL, m_chi_empty.getOutputDataNormalizer());
    EXPECT_EQ( NULL, m_chi_empty.getIntensityFunction());
    EXPECT_EQ( double(0), m_chi_empty.getValue());
    ASSERT_THROW(m_chi_empty.calculateChiSquared(), NullPointerException);
    ASSERT_THROW(m_chi_empty.createChi2DifferenceMap(), NullPointerException);
}

TEST_F(ChiSquaredModuleTest, CloneOfEmpty)
{
    ChiSquaredModule *clone_of_empty = m_chi_empty.clone();
    EXPECT_EQ( NULL, clone_of_empty->getRealData());
    EXPECT_EQ( NULL, clone_of_empty->getSimulationData());
    EXPECT_TRUE( dynamic_cast<const SquaredFunctionDefault *>(clone_of_empty->getSquaredFunction()));
    EXPECT_TRUE( dynamic_cast<const DefaultAllDataSelector *>(clone_of_empty->getFittingDataSelector()));
    EXPECT_EQ( NULL, clone_of_empty->getOutputDataNormalizer());
    EXPECT_EQ( NULL, clone_of_empty->getIntensityFunction());
    EXPECT_EQ( double(0), clone_of_empty->getValue());
    ASSERT_THROW(clone_of_empty->calculateChiSquared(), NullPointerException);
    ASSERT_THROW(clone_of_empty->createChi2DifferenceMap(), NullPointerException);
    delete clone_of_empty;
}

TEST_F(ChiSquaredModuleTest, DefaultModule)
{
    EXPECT_FLOAT_EQ( double(0.01), m_chi_default.calculateChiSquared());
    EXPECT_FLOAT_EQ( double(0.01), m_chi_default.getValue());
}

TEST_F(ChiSquaredModuleTest, CloneOfDefault)
{
    ChiSquaredModule *clone_of_default = m_chi_default.clone();
    EXPECT_FLOAT_EQ( double(0.01), clone_of_default->calculateChiSquared());
    EXPECT_FLOAT_EQ( double(0.01), clone_of_default->getValue());
    clone_of_default->setNdegreeOfFreedom(1);
    EXPECT_FLOAT_EQ( double(1.0), clone_of_default->calculateChiSquared());
    delete clone_of_default;
}


TEST_F(ChiSquaredModuleTest, IsGISAXSLikeModule)
{
    ChiSquaredModule chi_isgisaxs;
    OutputData<double > real_data;
    OutputData<double > simul_data;
    const size_t nbins(5);
    real_data.addAxis(NDetector2d::PHI_AXIS_NAME, nbins, 0.0, 1.0);
    simul_data.addAxis(NDetector2d::PHI_AXIS_NAME, nbins, 0.0, 1.0);
    const double a_real_data[nbins] = {1., 10., 100., 10., 1. };
    const double a_simul_data[nbins] = {10., 100., 1000., 100., 10. };
    OutputData<double >::iterator it_real = real_data.begin();
    OutputData<double >::iterator it_simul = simul_data.begin();
    int index(0);
    while(it_real != real_data.end()) {
        *it_real = a_real_data[index];
        *it_simul = a_simul_data[index];
        ++index; ++it_real; ++it_simul;
    }
    chi_isgisaxs.setRealAndSimulatedData(real_data, simul_data);
    OutputDataNormalizerScaleAndShift normalizer(100., 0.0);
    chi_isgisaxs.setOutputDataNormalizer( normalizer );
    EXPECT_FLOAT_EQ( double(0.0), chi_isgisaxs.calculateChiSquared());

//    m_chi_isgisaxs.setChiSquaredFunction( SquaredFunctionWithSystematicError(1.0) );
//    EXPECT_FLOAT_EQ( double(0.005), m_chi_isgisaxs.calculateChiSquared());

//    m_chi_isgisaxs.setOutputDataNormalizer( OutputDataNormalizerScaleAndShift() );
//    EXPECT_FLOAT_EQ( double(0.005), m_chi_isgisaxs.calculateChiSquared());


}


#endif // CHISQUAREDMODULETEST_H




