// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Computation/SpecularComputationTerm.cpp
//! @brief     Implements functor SpecularComputationTerm.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "SpecularComputationTerm.h"
#include "DelayedProgressCounter.h"
#include "ScalarRTCoefficients.h"
#include "SpecularScalarStrategy.h"
#include "SpecularSimulationElement.h"
#include <iostream>

SpecularComputationTerm::SpecularComputationTerm(std::unique_ptr<ISpecularStrategy> strategy)
    : m_Strategy(std::move(strategy)){};

SpecularScalarTerm::SpecularScalarTerm(std::unique_ptr<ISpecularStrategy> strategy)
    : SpecularComputationTerm(std::move(strategy))
{
}

SpecularComputationTerm::~SpecularComputationTerm() = default;

void SpecularComputationTerm::setProgressHandler(ProgressHandler* p_progress)
{
    mP_progress_counter.reset(new DelayedProgressCounter(p_progress, 100));
}

void SpecularComputationTerm::compute(SpecularSimulationElement& elem,
                                      const std::vector<Slice>& slices) const
{
    if (!elem.isCalculated())
        return;

    eval(elem, slices);

    if (mP_progress_counter)
        mP_progress_counter->stepProgress();
}

SpecularScalarTerm::~SpecularScalarTerm() = default;

void SpecularScalarTerm::eval(SpecularSimulationElement& elem,
                              const std::vector<Slice>& slices) const
{
    auto coeff = m_Strategy->Execute(slices, elem.produceKz(slices));
    elem.setIntensity(std::norm(coeff.front()->getScalarR()));
}

SpecularMatrixTerm::SpecularMatrixTerm(std::unique_ptr<ISpecularStrategy> strategy)
    : SpecularComputationTerm(std::move(strategy))
{
}

SpecularMatrixTerm::~SpecularMatrixTerm() = default;

void SpecularMatrixTerm::eval(SpecularSimulationElement& elem,
                              const std::vector<Slice>& slices) const
{
    auto coeff = m_Strategy->Execute(slices, elem.produceKz(slices));

//    for(size_t i = 0; i < coeff.size(); ++i)
    for(size_t i = 0; i < 1; ++i)
    {
        auto c = coeff[i].get();
        std::cout << "========================================\n";
        std::cout << "i = " << i << "\n";
//        c->T1plus();
//        c->T1min();

//        c->T1min();
        std::cout << "+\n";
        std::cout << "T1 = " << c->T1plus() << " T2 = " << c->T2plus() << "\n";
        std::cout << "R1 = " << c->R1plus() << " R2 = " << c->R2plus() << "\n";

        std::cout << "-\n";
        std::cout << "T1 = " << c->T1min() << " T2 = " << c->T2min() << "\n";
        std::cout << "R1 = " << c->R1min() << " R2 = " << c->R2min() << std::endl;


    }

    elem.setIntensity(intensity(elem, coeff.front()));
}

double SpecularMatrixTerm::intensity(const SpecularSimulationElement& elem,
                                     const ISpecularStrategy::single_coeff_t& coeff) const
{
    const auto& polarization = elem.polarizationHandler().getPolarization();
    const auto& analyzer = elem.polarizationHandler().getAnalyzerOperator();

    auto R = coeff->getReflectionMatrix();

    const complex_t result = (polarization * R.adjoint() * analyzer * R).trace();

    return std::abs(result);
}
