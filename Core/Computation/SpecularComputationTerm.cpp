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
    elem.setIntensity(intensity(elem, coeff.front()));
}

double SpecularMatrixTerm::intensity(const SpecularSimulationElement& elem,
                                     ISpecularStrategy::single_coeff_t& coeff) const
{
    const auto& polarization = elem.polarizationHandler().getPolarization();
    const auto& analyzer = elem.polarizationHandler().getAnalyzerOperator();

    // construct the reflection operator
//    auto M = coeff->getM();

//    std::cout << "M = " << M << std::endl;

//    auto && precFunc = [](const auto ML, const auto MM, const auto MS,
//                                auto i0, auto i1, auto j0, auto j1, auto k0, auto k1, auto l0, auto l1)
    auto && precFunc = [](const auto MM, const auto MS,
                                auto i0, auto i1, auto j0, auto j1, auto k0, auto k1, auto l0, auto l1)
    {
//        auto result = ML(i0, i1) * ML(j0, j1) - ML(k0, k1) * ML(l0, l1);
//        result += ML(i0, i1) * MM(j0, j1) + MM(i0, i1) * ML(j0, j1) + MM(i0, i1) * MM(j0, j1);
//        result -= (ML(k0, k1) * MM(l0, l1) + MM(k0, k1) * ML(l0, l1) + MM(k0, k1) * MM(l0, l1));
//        result += ML(i0, i1) * MS(j0, j1) + MS(i0, i1) * ML(j0, j1);
//        result -= ( ML(k0, k1) * MS(l0, l1) + MS(k0, k1) * ML(l0, l1) );

//        auto result = MM(i0, i1) * MM(j0, j1) - MM(k0, k1) * MM(l0, l1);
//            result += MM(i0, i1) * MS(j0, j1) + MS(i0, i1) * MM(j0, j1);
//            result -= (MM(k0, k1) * MS(l0, l1) + MS(k0, k1) * MM(l0, l1));
//            result += MS(i0, i1) * MS(j0, j1) - MS(k0, k1) * MS(l0, l1);

        // TODO: test or argue why MM(i0, i1) * MM(j0, j1) - MM(k0, k1) * MM(l0, l1);
        // always vanishes
        // including it, this becomes unstable
//        auto result = MM(i0, i1) * MM(j0, j1) - MM(k0, k1) * MM(l0, l1);
//            result += MM(i0, i1) * MS(j0, j1) + MS(i0, i1) * MM(j0, j1);
//            result -= (MM(k0, k1) * MS(l0, l1) + MS(k0, k1) * MM(l0, l1));
//            result += MS(i0, i1) * MS(j0, j1) - MS(k0, k1) * MS(l0, l1);

        auto diff = std::abs((MM(i0, i1) * MM(j0, j1) - MM(k0, k1) * MM(l0, l1))/(MM(k0, k1) * MM(l0, l1)));
        if ( !std::isnan(diff) && diff > 10 * std::numeric_limits<double>::epsilon() )
            throw std::runtime_error("Neglected part too large");

        auto result = MM(i0, i1) * MS(j0, j1) + MS(i0, i1) * MM(j0, j1);
            result -= (MM(k0, k1) * MS(l0, l1) + MS(k0, k1) * MM(l0, l1));
            result += MS(i0, i1) * MS(j0, j1) - MS(k0, k1) * MS(l0, l1);

        return result;
    };


    auto denominator = precFunc(coeff->getMM(), coeff->getMS(),
                                            0, 1,   1, 0,   0, 0,   1, 1);

//    std::cout << "denom2 = " << denominator2 << std::endl;

//    std::cout << "  rpp = " << M(2, 1) * M(1, 0) - M(2, 0) * M(1, 1) << std::endl;
//    std::cout << "  rmm = " << M(3, 0) * M(0, 1) - M(3, 1) * M(0, 0) << std::endl;

    Eigen::Matrix2cd R;
    R(0, 0) = precFunc(coeff->getMM(), coeff->getMS(),
                       2, 1,   1, 0,   2, 0,   1, 1);
    R(0, 1) = precFunc(coeff->getMM(), coeff->getMS(),
                        2, 0,   0, 1,   0, 0,   2, 1);
    R(1, 1) = precFunc(coeff->getMM(), coeff->getMS(),
                       3, 0,   0, 1,   3, 1,   0, 0);
    R(1, 0) = precFunc(coeff->getMM(), coeff->getMS(),
                       3, 1,   1, 0,   3, 0,   1, 1);

//    auto denominator = M(0,1) * M(1, 0) - M(0, 0) * M(1, 1);
//    R(0, 0) = M(2, 1) * M(1, 0) - M(2, 0) * M(1, 1);
//    R(0, 1) = M(2, 0) * M(0, 1) - M(0, 0) * M(2, 1);
//    R(1, 1) = M(3, 0) * M(0, 1) - M(3, 1) * M(0, 0);
//    R(1, 0) = M(3, 1) * M(1, 0) - M(3, 0) * M(1, 1);

//    std::cout << "denom  = " << denominator << std::endl;
//    std::cout << "   rpp = " << R(0, 0) << std::endl;
//    std::cout << "   rmm = " << R(1, 1) << std::endl;

    auto trickyDiv = [](auto M, const auto div)
    {
//      for(auto m : M.reshaped())
//      {
//        std::cout << "M = " << M << " div = " << div << std::endl;
          double max = std::max( std::abs(div.real()), std::abs(div.imag()) );
//          std::cout << "max = " << max << std::endl;
          auto divnorm = div / max;
//          std::cout << "divn = " << divnorm << std::endl;
          auto r = M/max;
//          std::cout << "r1 = " << r << std::endl;
          r /= divnorm;
          return r;
//      }

    };
//R.re
//    R /= denominator;
//    trickyDiv(R, denominator);
    R(0, 0) = trickyDiv(R(0, 0), denominator);
    R(0, 1) = trickyDiv(R(0, 1), denominator);
    R(1, 0) = trickyDiv(R(1, 0), denominator);
    R(1, 1) = trickyDiv(R(1, 1), denominator);

//    std::cout << "   R = " << R << std::endl;

//    std::cout << "pol = " << polarization << std::endl;
//    std::cout << "analyzer = " << analyzer << std::endl;

    const complex_t result = (polarization * R.adjoint() * analyzer * R).trace();
//    std::cout << " R = " << std::abs(result) << std::endl;
    return std::abs(result);
}
