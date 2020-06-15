// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Multilayer/SpecularMagneticNewStrategy.cpp
//! @brief     Implements class SpecularMagneticNewStrategy.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2020
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "SpecularMagneticNewStrategy.h"
#include "KzComputation.h"
#include "PhysicalConstants.h"
#include "Slice.h"

namespace
{
double magneticSLD(kvector_t B_field);
Eigen::Vector2cd eigenvalues(complex_t kz, double b_mag);
Eigen::Vector2cd checkForUnderflow(const Eigen::Vector2cd& eigenvs);
complex_t GetImExponential(complex_t exponent);

// The factor 1e-18 is here to have unit: 1/T*nm^-2
constexpr double magnetic_prefactor = PhysConsts::m_n * PhysConsts::g_factor_n * PhysConsts::mu_N
                                      / PhysConsts::h_bar / PhysConsts::h_bar / 4. / M_PI * 1e-18;
constexpr complex_t I(0.0, 1.0);


} // namespace

ISpecularStrategy::coeffs_t SpecularMagneticNewStrategy::Execute(const std::vector<Slice>& slices,
                                                              const kvector_t& k) const
{
    return Execute(slices, KzComputation::computeReducedKz(slices, k));
}

ISpecularStrategy::coeffs_t
SpecularMagneticNewStrategy::Execute(const std::vector<Slice>& slices,
                                  const std::vector<complex_t>& kz) const
{
    if (slices.size() != kz.size())
        throw std::runtime_error("Number of slices does not match the size of the kz-vector");

    ISpecularStrategy::coeffs_t result;
    for (auto& coeff : computeTR(slices, kz))
        result.push_back(std::make_unique<MatrixRTCoefficients_v3>(coeff));

    return result;
}

std::vector<MatrixRTCoefficients_v3>
SpecularMagneticNewStrategy::computeTR(const std::vector<Slice>& slices,
                                    const std::vector<complex_t>& kzs)
{
    if (slices.size() != kzs.size())
        throw std::runtime_error(
            "Error in SpecularMagnetic_::execute: kz vector and slices size shall coinside.");
    if (slices.empty())
        return {};

    std::vector<MatrixRTCoefficients_v3> result;
    result.reserve(slices.size());

    if( std::abs( kzs[0] ) < 10 * std::numeric_limits<double>::epsilon() )
    {
        for (size_t i = 0, size = slices.size(); i < size; ++i)
            result.emplace_back(0., Eigen::Vector2cd{0.0, 0.0}, kvector_t{0.0, 0.0, 0.0}, 0.0);

        result[0].m_MS.topLeftCorner(2, 2)     = Eigen::Matrix2cd::Identity(2, 2);
        result[0].m_MS.topRightCorner(2, 2)    = -Eigen::Matrix2cd::Identity(2, 2);
        result[0].m_MS.bottomLeftCorner(2, 2)  = -Eigen::Matrix2cd::Identity(2, 2);
        result[0].m_MS.bottomRightCorner(2, 2) = Eigen::Matrix2cd::Identity(2, 2);
        result[0].m_MS /= 2.;

        auto R = result[0].getReflectionMatrix();
        result[0].m_t_r_plus  << 1., 0., R(0, 0), R(1, 0);
        result[0].m_t_r_minus << 0., 1., R(0, 1), R(1, 1);

        return result;
    }

    const double kz_sign = kzs.front().real() > 0.0 ? 1.0 : -1.0; // save sign to restore it later
    auto B_0 = slices.front().bField();
    result.emplace_back(kz_sign, eigenvalues(kzs.front(), 0.0), kvector_t{0.0, 0.0, 0.0}, 0.0);
    for (size_t i = 1, size = slices.size(); i < size; ++i)
    {
        auto B = slices[i].bField() - B_0;
        auto magnetic_SLD = magneticSLD(B);
        result.emplace_back(kz_sign, checkForUnderflow(eigenvalues(kzs[i], magnetic_SLD)),
                            B.mag() > std::numeric_limits<double>::epsilon() * 10 ? B/B.mag() : kvector_t{0.0, 0.0, 0.0}, magnetic_SLD);
    }

    // calculate the matrices M_i
    for (size_t i = 0, interfaces = slices.size() - 1; i < interfaces; ++i) {


        auto mproduct = Eigen::Matrix2cd( result[i].computeInverseP() * result[i+1].computeP() );
        auto mp       = Eigen::Matrix2cd( Eigen::Matrix2cd::Identity() + mproduct );
        auto mm       = Eigen::Matrix2cd( Eigen::Matrix2cd::Identity() - mproduct );
        auto deltaTemp = result[i].computeDeltaMatrix(slices[i].thickness(), 1.);
        auto delta     = std::get<0>(deltaTemp) + std::get<1>(deltaTemp);
        auto deltaInv  = result[i].computeDeltaMatrix(slices[i].thickness(), -1.);

        result[i].m_MiL = Eigen::Matrix4cd::Zero();
        result[i].m_MiL.block<2,2>(0, 0) = std::get<0>(deltaInv) * mp;
        result[i].m_MiL.block<2,2>(0, 2) = std::get<0>(deltaInv) * mm;

        result[i].m_MiS = Eigen::Matrix4cd::Zero();
        result[i].m_MiS.block<2,2>(0, 0) = std::get<1>(deltaInv) * mp;
        result[i].m_MiS.block<2,2>(0, 2) = std::get<1>(deltaInv) * mm;
        result[i].m_MiS.block<2,2>(2, 0) = delta * mm;
        result[i].m_MiS.block<2,2>(2, 2) = delta * mp;

        result[i].m_MiL /= 2.;
        result[i].m_MiS /= 2.;

    }


    // calculate the total tranfer matrix M
    if(slices.size() == 2)
    {
        result[0].m_ML = result[0].m_MiL;
        result[0].m_MS = result[0].m_MiS;
    }
    else
    {
        result[slices.size()-2].m_ML = result[slices.size()-2].m_MiL;
        result[slices.size()-2].m_MS = result[slices.size()-2].m_MiS;
    }

    for (int i = slices.size() - 3; i >= 0; --i)
    {
        result[i].m_ML = result[i].m_MiL * result[i+1].m_ML + result[i].m_MiS * result[i+1].m_ML + result[i].m_MiL * result[i+1].m_MS;
        result[i].m_MS = result[i].m_MiS * result[i+1].m_MS;
    }

    // forward propagation
    // boundary condition in first layer
    auto R = result[0].getReflectionMatrix();
    result[0].m_t_r_plus  << 1., 0., R(0, 0), R(1, 0);
    result[0].m_t_r_minus << 0., 1., R(0, 1), R(1, 1);

    for(size_t i = 0, interfaces = slices.size() - 1; i < interfaces; ++i)
    {
        auto PInv = Eigen::Matrix2cd( result[i+1].computeInverseP() * result[i].computeP() );
        auto mp = Eigen::Matrix2cd::Identity() + PInv;
        auto mm = Eigen::Matrix2cd::Identity() - PInv;

        auto deltaTemp = result[i].computeDeltaMatrix(slices[i].thickness(), 1.);
        auto delta = std::get<0>(deltaTemp) + std::get<1>(deltaTemp);
        auto deltaInv = result[i].computeDeltaMatrix(slices[i].thickness(), -1.);


        Eigen::Matrix4cd MS{Eigen::Matrix4cd::Zero()};
        Eigen::Matrix4cd ML{Eigen::Matrix4cd::Zero()};

        ML.block<2,2>(0, 2) = std::get<0>(deltaInv) * mm;
        ML.block<2,2>(2, 2) = std::get<0>(deltaInv) * mp;

        MS.block<2,2>(0, 2) = std::get<1>(deltaInv) * mm;
        MS.block<2,2>(2, 2) = std::get<1>(deltaInv) * mp;
        MS.block<2,2>(2, 0) = delta * mm;
        MS.block<2,2>(0, 0) = delta * mp;

        MS /= 2.;
        ML /= 2.;

        result[i+1].m_t_r_plus  = ( MS + ML ) * result[i].m_t_r_plus;
        result[i+1].m_t_r_minus = ( MS + ML ) * result[i].m_t_r_minus;
    }

    return result;
}

namespace
{
double magneticSLD(kvector_t B_field)
{
    return magnetic_prefactor * B_field.mag();
}

Eigen::Vector2cd eigenvalues(complex_t kz, double magnetic_SLD)
{
    const complex_t a = kz * kz;
    return {std::sqrt(a - 4. * M_PI * magnetic_SLD), std::sqrt(a + 4. * M_PI * magnetic_SLD)};
}

Eigen::Vector2cd checkForUnderflow(const Eigen::Vector2cd& eigenvs)
{
    auto lambda = [](complex_t value) { return std::abs(value) < 1e-40 ? 1e-40 : value; };
    return {lambda(eigenvs(0)), lambda(eigenvs(1))};
}

// TODO: use this one? why used? branch cuts?
complex_t GetImExponential(complex_t exponent)
{
    if (exponent.imag() > -std::log(std::numeric_limits<double>::min()))
        return 0.0;
    return std::exp(I * exponent);
}
} // namespace
