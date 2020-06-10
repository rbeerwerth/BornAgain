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

namespace
{
Eigen::Vector2cd waveVector(const Eigen::Matrix4cd& frob_matrix,
                            const Eigen::Vector4cd& boundary_cond);

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

Eigen::Vector2cd MatrixRTCoefficients_v3::T1plus() const
{
    const Eigen::Vector2cd result = waveVector(T1, m_w_plus);
    if (m_lambda(0) == 0.0 && result == Eigen::Vector2cd::Zero())
        return {0.5, 0.0};
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::R1plus() const
{
    if (m_lambda(0) == 0.0 && waveVector(T1, m_w_plus) == Eigen::Vector2cd::Zero())
        return {-0.5, 0.0};
    return waveVector(R1, m_w_plus);
}

Eigen::Vector2cd MatrixRTCoefficients_v3::T2plus() const
{
    const Eigen::Vector2cd result = waveVector(T2, m_w_plus);
    if (m_lambda(1) == 0.0 && result == Eigen::Vector2cd::Zero())
        return {0.5, 0.0};
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::R2plus() const
{
    if (m_lambda(1) == 0.0 && waveVector(T2, m_w_plus) == Eigen::Vector2cd::Zero())
        return {-0.5, 0.0};
    return waveVector(R2, m_w_plus);
}

Eigen::Vector2cd MatrixRTCoefficients_v3::T1min() const
{
    const Eigen::Vector2cd result = waveVector(T1, m_w_min);
    if (m_lambda(0) == 0.0 && result == Eigen::Vector2cd::Zero())
        return {0.0, 0.5};
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::R1min() const
{
    if (m_lambda(0) == 0.0 && waveVector(T1, m_w_min) == Eigen::Vector2cd::Zero())
        return {0.0, -0.5};
    return waveVector(R1, m_w_min);
}

Eigen::Vector2cd MatrixRTCoefficients_v3::T2min() const
{
    const Eigen::Vector2cd result = waveVector(T2, m_w_min);
    if (m_lambda(1) == 0.0 && result == Eigen::Vector2cd::Zero())
        return {0.0, 0.5};
    return result;
}

Eigen::Vector2cd MatrixRTCoefficients_v3::R2min() const
{
    if (m_lambda(1) == 0.0 && waveVector(T2, m_w_min) == Eigen::Vector2cd::Zero())
        return {0.0, -0.5};
    return waveVector(R2, m_w_min);
}

Eigen::Vector2cd MatrixRTCoefficients_v3::getKz() const
{
    return -I * m_kz_sign * m_lambda;
}

Eigen::Matrix2cd MatrixRTCoefficients_v3::getReflectionMatrix() const
{
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

    auto trickyDiv = [](auto M, const auto div)
    {
          double max = std::max( std::abs(div.real()), std::abs(div.imag()) );
          auto divnorm = div / max;
          auto r = M/max;
          r /= divnorm;
          return r;
    };

    Eigen::Matrix2cd R;

    auto denominator = precFunc(getMM(), getMS(),
                                            0, 1,   1, 0,   0, 0,   1, 1);

    if( std::isinf(denominator.real()) || std::isinf(denominator.imag()) ||
            std::isnan(denominator.real()) || std::isinf(denominator.imag()) )
        throw std::runtime_error("Pushed this beyond numerical limits");

    R(0, 0) = precFunc(getMM(), getMS(),
                       2, 1,   1, 0,   2, 0,   1, 1);
    R(0, 1) = precFunc(getMM(), getMS(),
                        2, 0,   0, 1,   0, 0,   2, 1);
    R(1, 1) = precFunc(getMM(), getMS(),
                       3, 0,   0, 1,   3, 1,   0, 0);
    R(1, 0) = precFunc(getMM(), getMS(),
                       3, 1,   1, 0,   3, 0,   1, 1);

    R(0, 0) = trickyDiv(R(0, 0), denominator);
    R(0, 1) = trickyDiv(R(0, 1), denominator);
    R(1, 0) = trickyDiv(R(1, 0), denominator);
    R(1, 1) = trickyDiv(R(1, 1), denominator);

    return R;
}

namespace
{
Eigen::Vector2cd waveVector(const Eigen::Matrix4cd& frob_matrix,
                            const Eigen::Vector4cd& boundary_cond)
{
    Eigen::Matrix<complex_t, 4, 1> m = frob_matrix * boundary_cond;
    return {m(2), m(3)};
}
} // namespace
