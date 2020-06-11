// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file     Core/Multilayer/MatrixRTCoefficients_v3.cpp
//! @brief    Implements class MatrixRTCoefficients_v3.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "MatrixRTCoefficients_v3.h"
#include <iostream>

namespace
{
constexpr complex_t I = complex_t(0.0, 1.0);
} // namespace

MatrixRTCoefficients_v3::MatrixRTCoefficients_v3(double kz_sign, Eigen::Vector2cd eigenvalues,
                                                 kvector_t b, double magnetic_SLD)
    : m_kz_sign(kz_sign), m_lambda(std::move(eigenvalues)), m_b(std::move(b)), m_magnetic_SLD(magnetic_SLD)
{
}

MatrixRTCoefficients_v3::MatrixRTCoefficients_v3(const MatrixRTCoefficients_v3& other) = default;

MatrixRTCoefficients_v3::~MatrixRTCoefficients_v3() = default;

MatrixRTCoefficients_v3* MatrixRTCoefficients_v3::clone() const
{
    return new MatrixRTCoefficients_v3(*this);
}

Eigen::Matrix2cd MatrixRTCoefficients_v3::T1Matrix() const
{
    auto b = m_b;

    Eigen::Matrix2cd result;
    Eigen::Matrix2cd Q;

    auto factor1 = std::sqrt(2. * ( 1. + b.z()));
    auto factor2 = std::sqrt(2. * ( 1. - b.z()));

    Q << (1. + b.z()) / factor1, (b.z() - 1.) / factor2,
            (b.x() + I * b.y()) / factor1, (b.x() + I * b.y()) / factor2;

    auto exp2 = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({0., 1.}) );

    if ( std::abs(b.mag() - 1.) < std::numeric_limits<double>::epsilon() * 10.)
        result = Q * exp2 * Q.adjoint();
    else if(b.mag() == 0. && m_lambda(1) != 0.)
        result = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({0., 1.}) );
    else if( b.mag() == 0. && m_lambda(1) == 0. )
        result = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({0.5, 0.5}) );
    else
        throw std::runtime_error("Broken magnetic field vector");

    return result;
}

Eigen::Matrix2cd MatrixRTCoefficients_v3::T2Matrix() const
{
    auto b = m_b;

    Eigen::Matrix2cd result;
    Eigen::Matrix2cd Q;

    auto factor1 = std::sqrt(2. * ( 1. + b.z()));
    auto factor2 = std::sqrt(2. * ( 1. - b.z()));

    Q << (1. + b.z()) / factor1, (b.z() - 1.) / factor2,
            (b.x() + I * b.y()) / factor1, (b.x() + I * b.y()) / factor2;

    auto exp2 = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({1., 0.}) );

    if ( std::abs(b.mag() - 1.) < std::numeric_limits<double>::epsilon() * 10.)
        result = Q * exp2 * Q.adjoint();
    else if( b.mag() == 0. && m_lambda(0) != 0. )
        result = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({1., 0.}) );
    else if( b.mag() == 0. && m_lambda(0) == 0. )
        result = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({0.5, 0.5}) );
    else
        throw std::runtime_error("Broken magnetic field vector");

    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::T1plus() const
{
    auto mat = T1Matrix();
    auto redvec = Eigen::Vector2cd{ m_t_r_plus(0), m_t_r_plus(1) };
    auto result = mat * redvec;
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::R1plus() const
{
    auto mat = T1Matrix();
    auto redvec = Eigen::Vector2cd{ m_t_r_plus(2), m_t_r_plus(3) };
    auto result = mat * redvec;
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::T2plus() const
{
    auto mat = T2Matrix();
    auto redvec = Eigen::Vector2cd{ m_t_r_plus(0), m_t_r_plus(1) };
    auto result = mat * redvec;
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::R2plus() const
{
    auto mat = T2Matrix();
    auto redvec = Eigen::Vector2cd{ m_t_r_plus(2), m_t_r_plus(3) };
    auto result = mat * redvec;
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::T1min() const
{
    auto mat = T1Matrix();
    auto redvec = Eigen::Vector2cd{ m_t_r_minus(0), m_t_r_minus(1) };
    auto result = mat * redvec;
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::R1min() const
{
    auto mat = T1Matrix();
    auto redvec = Eigen::Vector2cd{ m_t_r_minus(2), m_t_r_minus(3) };
    auto result = mat * redvec;
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::T2min() const
{
    auto mat = T2Matrix();
    auto redvec = Eigen::Vector2cd{ m_t_r_minus(0), m_t_r_minus(1) };
    auto result = mat * redvec;
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::R2min() const
{
    auto mat = T2Matrix();
    auto redvec = Eigen::Vector2cd{ m_t_r_minus(2), m_t_r_minus(3) };
    auto result = mat * redvec;
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::getKz() const
{
    return m_kz_sign * m_lambda;
}

Eigen::Matrix2cd MatrixRTCoefficients_v3::getReflectionMatrix() const
{
    auto && precFunc = [](const auto & ML, const auto & MS,
                                auto i0, auto i1, auto j0, auto j1, auto k0, auto k1, auto l0, auto l1)
    {

        auto diff = std::abs((ML(i0, i1) * ML(j0, j1) - ML(k0, k1) * ML(l0, l1))/(ML(k0, k1) * ML(l0, l1)));
        if ( !std::isnan(diff) && diff > 10 * std::numeric_limits<double>::epsilon() )
            throw std::runtime_error("Neglected part too large");

        auto result = ML(i0, i1) * MS(j0, j1) + MS(i0, i1) * ML(j0, j1);
            result -= (ML(k0, k1) * MS(l0, l1) + MS(k0, k1) * ML(l0, l1));
            result += MS(i0, i1) * MS(j0, j1) - MS(k0, k1) * MS(l0, l1);

        return result;
    };

    auto trickyDiv = [](auto M, const auto div)
    {
          double max = std::max( std::abs(div.real()), std::abs(div.imag()) );
          auto divnorm = div / max;
          auto r = M/max;
          r /= divnorm;
          return r;
    };

    Eigen::Matrix2cd R;

    auto denominator = precFunc(m_ML, m_MS, 0, 1,   1, 0,   0, 0,   1, 1);

    if( std::isinf(denominator.real()) || std::isinf(denominator.imag()) ||
            std::isnan(denominator.real()) || std::isinf(denominator.imag()) )
        throw std::runtime_error("Pushed this beyond numerical limits");

    R(0, 0) = precFunc(m_ML, m_MS, 2, 1,   1, 0,   2, 0,   1, 1);
    R(0, 1) = precFunc(m_ML, m_MS, 2, 0,   0, 1,   0, 0,   2, 1);
    R(1, 1) = precFunc(m_ML, m_MS, 3, 0,   0, 1,   3, 1,   0, 0);
    R(1, 0) = precFunc(m_ML, m_MS, 3, 1,   1, 0,   3, 0,   1, 1);

    R(0, 0) = trickyDiv(R(0, 0), denominator);
    R(0, 1) = trickyDiv(R(0, 1), denominator);
    R(1, 0) = trickyDiv(R(1, 0), denominator);
    R(1, 1) = trickyDiv(R(1, 1), denominator);

    return R;
}
