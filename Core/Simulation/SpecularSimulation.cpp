// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Simulation/SpecularSimulation.cpp
//! @brief     Implements class OffSpecSimulation.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2015
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   C. Durniak, M. Ganeva, G. Pospelov, W. Van Herck, J. Wuttke
//
// ************************************************************************** //

#include "SpecularSimulation.h"
#include "IMultiLayerBuilder.h"
#include "MultiLayer.h"
#include "SpecularMatrix.h"
#include "MaterialUtils.h"
#include "Histogram1D.h"
#include <memory>

SpecularSimulation::SpecularSimulation() : Simulation()
{
    initialize();
}

SpecularSimulation::SpecularSimulation(const MultiLayer& sample) : Simulation(sample)
{
    initialize();
}

SpecularSimulation::SpecularSimulation(const std::shared_ptr<IMultiLayerBuilder> sample_builder)
    : Simulation(sample_builder)
{
    initialize();
}

SpecularSimulation::SpecularSimulation(const SpecularSimulation& other)
    : Simulation(other)
{
    if (other.m_alpha_i_axis)
	m_alpha_i_axis.reset(other.m_alpha_i_axis->clone());
    m_RT_coefficients.copyFrom(other.m_RT_coefficients);
    initialize();
}

SpecularSimulation* SpecularSimulation::clone() const
{
    return new SpecularSimulation(*this);
}

void SpecularSimulation::prepareSimulation()
{
    if (!m_alpha_i_axis || m_alpha_i_axis->size() < 1)
        throw std::runtime_error("Error in SpecularSimulation::prepareSimulation(): "
                                 "Incoming alpha range not configured.");

    Simulation::prepareSimulation();
    m_RT_coefficients.clear();
    m_RT_coefficients.addAxis(*m_alpha_i_axis);
}

void SpecularSimulation::runSimulation()
{
    prepareSimulation();

    const MultiLayer* multilayer = m_sample_provider.sample();
    multilayer->requiresMatrixRTCoefficients() ? collectRTCoefficientsMatrix(multilayer)
                                               : collectRTCoefficientsScalar(multilayer);
}

size_t SpecularSimulation::numberOfSimulationElements() const
{
    if (!m_alpha_i_axis)
        std::runtime_error("Error in SpecularSimulation::numberOfSimulationElements:"
                           "angle range was not initialized.");
    return m_alpha_i_axis->size();
}

void SpecularSimulation::setBeamParameters(double lambda, const IAxis& alpha_axis, double phi_i)
{
    if (alpha_axis.size() < 1)
        throw std::runtime_error("Error in SpecularSimulation::setBeamParameters(): "
                                 "incoming angle range size < 1.");
    m_alpha_i_axis.reset(alpha_axis.clone());
    m_instrument.setBeamParameters(lambda, alpha_axis[0], phi_i);
}

void SpecularSimulation::setBeamParameters(double lambda, int nbins, double alpha_i_min,
                                           double alpha_i_max, double phi_i)
{
    FixedBinAxis axis("alpha_i", nbins, alpha_i_min, alpha_i_max);
    setBeamParameters(lambda, axis, phi_i);
}

const IAxis* SpecularSimulation::getAlphaAxis() const
{
    return m_alpha_i_axis.get();
}

std::unique_ptr<OutputData<double>>
SpecularSimulation::getData(size_t i_layer, DataGetter fn_ptr) const
{
    checkCoefficients(i_layer);
    std::unique_ptr<OutputData<double>> output_ptr = std::make_unique<OutputData<double>>();
    output_ptr->copyShapeFrom(m_RT_coefficients);
    OutputData<double>& output = *output_ptr.get();
    for (size_t i = 0; i < m_RT_coefficients.getAllocatedSize(); ++i)
        output[i] = std::norm((m_RT_coefficients[i][i_layer].get()->*fn_ptr)());
    return output_ptr;
}

OutputData<double>* SpecularSimulation::getDetectorIntensity(DetectorAxesUnits) const
{
    return getData(0, &ILayerRTCoefficients::getScalarR).release();
}

Histogram1D* SpecularSimulation::reflectivity() const
{
    return new Histogram1D(*getData(0, &ILayerRTCoefficients::getScalarR).get());
}

Histogram1D* SpecularSimulation::transmissivity() const
{
    const size_t i_layer = m_RT_coefficients.isInitialized() && m_RT_coefficients[0].size() != 0
                               ? static_cast<size_t>(m_RT_coefficients[0].size() - 1)
                               : 0;
    return new Histogram1D(*getData(i_layer, &ILayerRTCoefficients::getScalarT).get());
}

std::vector<complex_t> SpecularSimulation::getScalarR(size_t i_layer) const
{
    checkCoefficients(i_layer);
    std::vector<complex_t> result;
    result.resize(m_RT_coefficients.getAllocatedSize());
    for (size_t i = 0; i < m_RT_coefficients.getAllocatedSize(); ++i) {
        result[i] = m_RT_coefficients[i][i_layer]->getScalarR();
    }
    return result;
}

std::vector<complex_t> SpecularSimulation::getScalarT(size_t i_layer) const
{
    checkCoefficients(i_layer);
    std::vector<complex_t> result;
    result.resize(m_RT_coefficients.getAllocatedSize());
    for (size_t i = 0; i < m_RT_coefficients.getAllocatedSize(); ++i) {
        result[i] = m_RT_coefficients[i][i_layer]->getScalarT();
    }
    return result;
}

std::vector<complex_t> SpecularSimulation::getScalarKz(size_t i_layer) const
{
    checkCoefficients(i_layer);
    std::vector<complex_t> result;
    result.resize(m_RT_coefficients.getAllocatedSize());
    for (size_t i = 0; i < m_RT_coefficients.getAllocatedSize(); ++i) {
        result[i] = m_RT_coefficients[i][i_layer]->getScalarKz();
    }
    return result;
}

void SpecularSimulation::collectRTCoefficientsScalar(const MultiLayer *multilayer)
{
    OutputData<MultiLayerRTCoefficients_t>::iterator it = m_RT_coefficients.begin();
    while (it != m_RT_coefficients.end()) {
        double alpha_i = m_RT_coefficients.getAxisValue(it.getIndex(), 0);
        kvector_t kvec = vecOfLambdaAlphaPhi(m_instrument.getBeam().getWavelength(), -alpha_i, 0.0);

        std::vector<ScalarRTCoefficients> coeffs;
        SpecularMatrix::execute(*multilayer, kvec, coeffs);

        MultiLayerRTCoefficients_t ml_coeffs;
        ml_coeffs.resize(coeffs.size());
        for (size_t i_layer = 0; i_layer < coeffs.size(); ++i_layer) {
            ml_coeffs[i_layer] = LayerRTCoefficients_t(new ScalarRTCoefficients(coeffs[i_layer]));
        }

        *it = ml_coeffs;
        ++it;

    } // alpha_i
}

void SpecularSimulation::collectRTCoefficientsMatrix(const MultiLayer* /*multilayer*/)
{
    throw std::runtime_error(
        "SpecularSimulation::collectRTCoefficientsMatrix() -> Error. Not implemented.");
}

void SpecularSimulation::checkCoefficients(size_t i_layer) const
{
    if (!m_RT_coefficients.isInitialized() || m_RT_coefficients[0].size() == 0)
        throw std::runtime_error(
            "SpecularSimulation::checkCoefficients() -> Error. "
            "No coefficients found, check that (1) you have set beam parameters "
            "(2) you have run your simulation.");

    if (i_layer >= m_RT_coefficients[0].size()) {
        std::ostringstream message;
        message << "SpecularSimulation::checkCoefficients() -> Error. Requested layer index "
                << i_layer << " is large than or equal to the total number of layers "
                << m_RT_coefficients[0].size() << std::endl;
        throw std::runtime_error(message.str());
    }
}

void SpecularSimulation::initialize()
{
    setName(BornAgain::SpecularSimulationType);
}
