// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file     Core/Multilayer/MatrixRTCoefficients_v3.cpp
//! @brief    Implements class MatrixRTCoefficients_v3.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2020
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "MatrixRTCoefficients_v3.h"

namespace
{
using matrixType = Eigen::Matrix4cd;

constexpr complex_t I = complex_t(0.0, 1.0);
complex_t elementProductDifference(const matrixType & ML, const matrixType & MS,
                   size_t i0, size_t i1, size_t j0, size_t j1, size_t k0, size_t k1, size_t l0, size_t l1);
complex_t complexDivision(const complex_t v, const complex_t div);
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

Eigen::Matrix2cd MatrixRTCoefficients_v3::TransformationMatrix(complex_t eigenvalue, Eigen::Vector2d selection) const
{
    Eigen::Matrix2cd result;
    Eigen::Matrix2cd Q;

    auto factor1 = std::sqrt(2. * ( 1. + m_b.z()));
    auto factor2 = std::sqrt(2. * ( 1. - m_b.z()));

    Q << (1. + m_b.z()) / factor1, (m_b.z() - 1.) / factor2,
            (m_b.x() + I * m_b.y()) / factor1, (m_b.x() + I * m_b.y()) / factor2;

    auto exp2 = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>(selection) );

    if ( std::abs(m_b.mag() - 1.) < std::numeric_limits<double>::epsilon() * 10.)
        result = Q * exp2 * Q.adjoint();
    else if( m_b.mag() == 0. && eigenvalue != 0. )
        result = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>(selection) );
    else if( m_b.mag() == 0. && eigenvalue == 0. )
        result = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({0.5, 0.5}) );
    else
        throw std::runtime_error("Broken magnetic field vector");

    return result;
}

Eigen::Matrix2cd MatrixRTCoefficients_v3::T1Matrix() const
{
    return TransformationMatrix(m_lambda(1), {0., 1.});
}

Eigen::Matrix2cd MatrixRTCoefficients_v3::T2Matrix() const
{
    return TransformationMatrix(m_lambda(0), {1., 0.});
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

Eigen::Matrix2cd MatrixRTCoefficients_v3::pMatrixHelper(double sign) const
{
    auto Lp = m_lambda(1) + m_lambda(0);
    auto Lm = m_lambda(1) - m_lambda(0);

    Eigen::Matrix2cd result;

    auto b = m_b;

    result << Lp + sign * Lm * b.z(), sign * Lm * ( b.x() - I * b.y() ),
              sign * Lm * ( b.x() + I * b.y() ), Lp - sign * Lm * b.z();

    return result;
}

Eigen::Matrix2cd MatrixRTCoefficients_v3::computeP() const
{
    auto result = pMatrixHelper(1.);
    result *= 0.5;

    return result;
}



Eigen::Matrix2cd MatrixRTCoefficients_v3::computeInverseP() const
{
    auto Lp = m_lambda(1) + m_lambda(0);
    auto Lm = m_lambda(1) - m_lambda(0);

    auto result = pMatrixHelper(-1.);
    result *= 2./(Lp * Lp - Lm * Lm);

    return result;
}



Eigen::Matrix2cd MatrixRTCoefficients_v3::getReflectionMatrix() const
{
    Eigen::Matrix2cd R;

    auto denominator = elementProductDifference(m_ML, m_MS, 0, 1,   1, 0,   0, 0,   1, 1);

    if( std::isinf(denominator.real()) || std::isinf(denominator.imag()) ||
            std::isnan(denominator.real()) || std::isinf(denominator.imag()) )
        throw std::runtime_error("Pushed this beyond numerical limits");

    R(0, 0) = elementProductDifference(m_ML, m_MS, 2, 1,   1, 0,   2, 0,   1, 1);
    R(0, 1) = elementProductDifference(m_ML, m_MS, 2, 0,   0, 1,   0, 0,   2, 1);
    R(1, 1) = elementProductDifference(m_ML, m_MS, 3, 0,   0, 1,   3, 1,   0, 0);
    R(1, 0) = elementProductDifference(m_ML, m_MS, 3, 1,   1, 0,   3, 0,   1, 1);

    R(0, 0) = complexDivision(R(0, 0), denominator);
    R(0, 1) = complexDivision(R(0, 1), denominator);
    R(1, 0) = complexDivision(R(1, 0), denominator);
    R(1, 1) = complexDivision(R(1, 1), denominator);

    return R;
}

namespace
{
complex_t elementProductDifference(const matrixType & ML, const matrixType & MS,
                   size_t i0, size_t i1, size_t j0, size_t j1, size_t k0, size_t k1, size_t l0, size_t l1)
{
    auto diff = std::abs((ML(i0, i1) * ML(j0, j1) - ML(k0, k1) * ML(l0, l1))/(ML(k0, k1) * ML(l0, l1)));
    if ( !std::isnan(diff) && diff > 10 * std::numeric_limits<double>::epsilon() )
        throw std::runtime_error("Neglected part too large");

    auto result = ML(i0, i1) * MS(j0, j1) + MS(i0, i1) * ML(j0, j1);
        result -= (ML(k0, k1) * MS(l0, l1) + MS(k0, k1) * ML(l0, l1));
        result += MS(i0, i1) * MS(j0, j1) - MS(k0, k1) * MS(l0, l1);

    return result;
};

complex_t complexDivision(const complex_t v, const complex_t div)
{
      double max = std::max( std::abs(div.real()), std::abs(div.imag()) );
      auto divnorm = div / max;
      auto r = v/max;
      r /= divnorm;
      return r;
};

}

