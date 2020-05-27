// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Multilayer/SpecularMagneticNewStrategy.cpp
//! @brief     Implements class SpecularMagneticNewStrategy.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "SpecularMagneticNewStrategy.h"
#include "KzComputation.h"
#include "PhysicalConstants.h"
#include "Slice.h"
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

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

Eigen::Matrix2cd SpecularMagneticNewStrategy::computeP(MatrixRTCoefficients_v3& coeff)
{
    auto Lp = coeff.m_lambda(0) + coeff.m_lambda(1);
    auto Lm = coeff.m_lambda(0) - coeff.m_lambda(1);

    Eigen::Matrix2cd result;

    auto b = coeff.m_b;

    result << Lp + Lm * b.z(), Lm * ( b.x() - I * b.y() ),
              Lm * ( b.x() + I * b.y() ), Lp - Lm * b.z();
    result *= 0.5;

    return result;
}

Eigen::Matrix2cd SpecularMagneticNewStrategy::computeInverseP(MatrixRTCoefficients_v3& coeff)
{
    auto Lp = coeff.m_lambda(0) + coeff.m_lambda(1);
    auto Lm = coeff.m_lambda(0) - coeff.m_lambda(1);

    Eigen::Matrix2cd result;

    auto b = coeff.m_b;

    result << Lp - Lm * b.z(), -Lm * ( b.x() - I * b.y() ),
              -Lm * ( b.x() + I * b.y() ), Lp + Lm * b.z();
    result *= 2./(Lp * Lp - Lm * Lm);

    return result;
}


// auto && cmpfct = [](const auto & cp1, const auto & cp2 ){ return std::norm(cp1) < std::norm(cp2); };
// auto && maxExp = std::max( {p1, p2, p3, p4}, cmpfct );



Eigen::Matrix2cd SpecularMagneticNewStrategy::computeDelta(MatrixRTCoefficients_v3& coeff, double thickness, double prefactor)
{
    auto b = coeff.m_b;

    std::cout << "b = " << b << std::endl;

    Eigen::Matrix2cd result;
    auto Lp = prefactor * I * 0.5 * thickness * (coeff.m_lambda(0) + coeff.m_lambda(1));
    auto Lm = prefactor * I * 0.5 * thickness * (coeff.m_lambda(0) - coeff.m_lambda(1));


    auto det = std::exp(Lp) * std::exp(Lp) * std::exp(Lm) * std::exp(-Lm);

    std::cout << "det(Delta) (in: computeDelta) = " << det << std::endl;

//    auto Q = Eigen::Matrix2cd::Zero();
    Eigen::Matrix2cd Q;

    auto factor1 = std::sqrt(2. * ( 1. + b.z()));
    auto factor2 = std::sqrt(2. * ( 1. - b.z()));

    Q << (1. + b.z()) / factor1, (b.z() - 1.) / factor2,
            (b.x() + I * b.y()) / factor1, (b.x() + I * b.y()) / factor2;

//    std::cout << "Q.+ * Q = " << Q.adjoint() * Q << std::endl;
//    Eigen::Matrix2cd exp2;
//    return result;



    auto exp1 = Eigen::DiagonalMatrix<complex_t, 2>({std::exp(Lp), std::exp(Lp)});
    auto exp2 = Eigen::DiagonalMatrix<complex_t, 2>({std::exp(Lm), std::exp(-Lm)});

//    Eigen::Matrix2cd result2

//    result = exp2 * Q * exp1 * Q.adjoint();
//    result = Q * Q.adjoint();
    if (b.mag() == 1.)
        result = exp1 * Q * exp2 * Q.adjoint();
    else if(b.mag() == 0.)
        result = Eigen::Matrix2cd(exp1) * Eigen::Matrix2cd(exp2);
    else
        throw std::runtime_error("Broken magnetic field vector");
//    result = exp1;
//    result = Q * exp2 * Q.adjoint();
//    std::cout << "Q = " << Q << std::endl;
    std::cout << "exp1 = " << Eigen::Matrix2cd( exp1 ) << std::endl;
    std::cout << "exp2 = " << Eigen::Matrix2cd( exp2 ) << std::endl;
    std::cout << "delta = " << result << std::endl;

//    return result;
    // lazy way
//    auto pm = prefactor * I * thickness * computeP(coeff);

//    auto expmatrix = pm.exp();
//    std::cout << "delta 2 = " << expmatrix << std::endl;
//    return expmatrix;
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

    const double kz_sign = kzs.front().real() > 0.0 ? 1.0 : -1.0; // save sign to restore it later
    auto B_0 = slices.front().bField();
    result.emplace_back(kz_sign, eigenvalues(kzs.front(), 0.0), kvector_t{0.0, 0.0, 0.0}, 0.0);
    for (size_t i = 1, size = slices.size(); i < size; ++i) {
        auto B = slices[i].bField() - B_0;
        auto magnetic_SLD = magneticSLD(B);
        result.emplace_back(kz_sign, checkForUnderflow(eigenvalues(kzs[i], magnetic_SLD)),
                            B.mag() != 0. ? B/B.mag() : kvector_t{0.0, 0.0, 0.0}, magnetic_SLD);
    }

    auto testindex{1};

    auto bi = result[testindex].m_b;

//    std::cout << "b = " << bi << std::endl;

    if(testindex >= slices.size())
        throw  std::runtime_error("Trying to access invalid slice");


    auto pm0  = computeP(result[testindex]);
    auto pmi0 = computeInverseP(result[testindex]);

//    std::cout << "test" << std::endl;
//    std::cout << "pm0 = " << pm0 << std::endl;
//    std::cout << "pmi0 = " << pmi0 << std::endl;


//    std::cout << "pm0 * pmi0 = " << pm0 * pmi0 << std::endl;


    // calculate the matrices M_i
    for (size_t i = 0, interfaces = slices.size() - 1; i < interfaces; ++i) {

        std::cout << "===================\ni = "<< i << std::endl;

        std::cout << "lp = " << result[i].m_lambda(0) << " lm = " << result[i].m_lambda(1) << std::endl;
        std::cout << "Lp = " << result[i].m_lambda(0) + result[i].m_lambda(1) << " Lm = " << result[i].m_lambda(0) - result[i].m_lambda(1) << std::endl;
        std::cout << "exponentials: " << std::exp(I * slices[i].thickness() * result[i].m_lambda(0) ) << " " <<
                                         std::exp(-I * slices[i].thickness() * result[i].m_lambda(0) ) << " " <<
                                         std::exp(I * slices[i].thickness() * result[i].m_lambda(1) ) << " " <<
                                         std::exp(-I * slices[i].thickness() * result[i].m_lambda(1) ) << std::endl;

        auto mproduct = computeInverseP(result[i]) * computeP(result[i+1]);
        auto mp = Eigen::Matrix2cd::Identity() + mproduct;
        auto mm = Eigen::Matrix2cd::Identity() - mproduct;
//        auto pm1 = ;
        auto delta = computeDelta(result[i], slices[i].thickness(), 1.);
        auto deltaInv = computeDelta(result[i], slices[i].thickness(), -1.);
//        auto Mi = Eigen::Matrix4cd::Zero();
//        std::cout << "delta = " << delta << std::endl;
//        std::cout << "delta^* = " << deltaInv << std::endl;

        std::cout << "det(delta) = " << delta(0, 0) * delta(1, 1) - delta(1, 0) * delta(0, 1) << std::endl;
        std::cout << "det(delta^*) = " << deltaInv(0, 0) * deltaInv(1, 1) - deltaInv(1, 0) * deltaInv(0, 1) << std::endl;


        result[i].Mi = Eigen::Matrix4cd::Zero();
        result[i].Mi.block<2,2>(0, 0) = deltaInv * mp;
        result[i].Mi.block<2,2>(0, 2) = deltaInv * mm;
        result[i].Mi.block<2,2>(2, 0) = delta * mm;
        result[i].Mi.block<2,2>(2, 2) = delta * mp;


//        result[i].Mi.block<2,2>(0, 0) = mp;
//        result[i].Mi.block<2,2>(0, 2) = mm;
//        result[i].Mi.block<2,2>(2, 0) = mm;
//        result[i].Mi.block<2,2>(2, 2) = mp;


        result[i].Mi /= 0.5;

//        std::cout << "i = " << i << std::endl;
        std::cout << "Mi = " << result[i].Mi << std::endl;

        std::cout << "det(Mi) = " << result[i].Mi(0, 1) * result[i].Mi(1, 0) - result[i].Mi(1, 1) * result[i].Mi(0, 0) << std::endl;

    }


    // calculate the total tranfer matrix M

    if(slices.size() == 2)
        result[0].M = result[0].Mi;

    else
        result[slices.size()-2].M = result[slices.size()-2].Mi;

    for (int i = slices.size() - 3; i >= 0; --i)
        result[i].M = result[i].Mi * result[i+1].M;


    std::cout << "M = " << result.front().M << std::endl;

    // extract R






    return result;

    // old stuff from here

//    if (result.front().m_lambda == Eigen::Vector2cd::Zero()) {
//        std::for_each(result.begin(), result.end(), [](auto& coeff) { setNoTransmission(coeff); });
//        return result;
//    }

//    std::for_each(result.begin(), result.end(), [](auto& coeff) { calculateTR(coeff); });
//    nullifyBottomReflection(result.back());
//    propagateBackwards(result, slices);
//    propagateForwards(result, findNormalizationCoefficients(result.front()));

//    return result;
}

void SpecularMagneticNewStrategy::calculateTR(MatrixRTCoefficients_v3& coeff)
{
    const double b = coeff.m_b.mag();
    if (b == 0.0) {
        calculateZeroFieldTR(coeff);
        return;
    }

    const double bpbz = b + coeff.m_b.z();
    const double bmbz = b - coeff.m_b.z();
    const complex_t bxmby = coeff.m_b.x() - I * coeff.m_b.y();
    const complex_t bxpby = coeff.m_b.x() + I * coeff.m_b.y();
    const complex_t l_1 = coeff.m_lambda(0);
    const complex_t l_2 = coeff.m_lambda(1);

    auto& T1 = coeff.T1;
    T1 << bmbz, -bxmby, -bmbz * l_1, bxmby * l_1, -bxpby, bpbz, bxpby * l_1, -bpbz * l_1,
        -bmbz / l_1, bxmby / l_1, bmbz, -bxmby, bxpby / l_1, -bpbz / l_1, -bxpby, bpbz;
    T1 /= 4.0 * b;

    auto& R1 = coeff.R1;
    R1 << T1(0, 0), T1(0, 1), -T1(0, 2), -T1(0, 3), T1(1, 0), T1(1, 1), -T1(1, 2), -T1(1, 3),
        -T1(2, 0), -T1(2, 1), T1(2, 2), T1(2, 3), -T1(3, 0), -T1(3, 1), T1(3, 2), T1(3, 3);

    auto& T2 = coeff.T2;
    T2 << bpbz, bxmby, -bpbz * l_2, -bxmby * l_2, bxpby, bmbz, -bxpby * l_2, -bmbz * l_2,
        -bpbz / l_2, -bxmby / l_2, bpbz, bxmby, -bxpby / l_2, -bmbz / l_2, bxpby, bmbz;
    T2 /= 4.0 * b;

    auto& R2 = coeff.R2;
    R2 << T2(0, 0), T2(0, 1), -T2(0, 2), -T2(0, 3), T2(1, 0), T2(1, 1), -T2(1, 2), -T2(1, 3),
        -T2(2, 0), -T2(2, 1), T2(2, 2), T2(2, 3), -T2(3, 0), -T2(3, 1), T2(3, 2), T2(3, 3);
}

void SpecularMagneticNewStrategy::calculateZeroFieldTR(MatrixRTCoefficients_v3& coeff)
{
    coeff.T1 = Eigen::Matrix4cd::Zero();
    coeff.R1 = Eigen::Matrix4cd::Zero();
    coeff.T2 = Eigen::Matrix4cd::Zero();
    coeff.R2 = Eigen::Matrix4cd::Zero();

    // lambda_1 == lambda_2, no difference which one to use
    const complex_t eigen_value = coeff.m_lambda(0);

    Eigen::Matrix3cd Tblock;
    Tblock << 0.5, 0.0, -0.5 * eigen_value, 0.0, 0.0, 0.0, -0.5 / eigen_value, 0.0, 0.5;

    Eigen::Matrix3cd Rblock;
    Rblock << 0.5, 0.0, 0.5 * eigen_value, 0.0, 0.0, 0.0, 0.5 / eigen_value, 0.0, 0.5;

    coeff.T1.block<3, 3>(1, 1) = Tblock;
    coeff.R1.block<3, 3>(1, 1) = Rblock;
    coeff.T2.block<3, 3>(0, 0) = Tblock;
    coeff.R2.block<3, 3>(0, 0) = Rblock;
}

void SpecularMagneticNewStrategy::setNoTransmission(MatrixRTCoefficients_v3& coeff)
{
    coeff.m_w_plus = Eigen::Vector4cd::Zero();
    coeff.m_w_min = Eigen::Vector4cd::Zero();
    coeff.T1 = Eigen::Matrix4cd::Identity() / 4.0;
    coeff.R1 = coeff.T1;
    coeff.T2 = coeff.T1;
    coeff.R2 = coeff.T1;
}

void SpecularMagneticNewStrategy::nullifyBottomReflection(MatrixRTCoefficients_v3& coeff)
{
    const complex_t l_1 = coeff.m_lambda(0);
    const complex_t l_2 = coeff.m_lambda(1);
    const double b_mag = coeff.m_b.mag();
    const kvector_t& b = coeff.m_b;

    if (b_mag == 0.0) {
        // both eigenvalues are the same, no difference which one to take
        coeff.m_w_plus << -l_1, 0.0, 1.0, 0.0;
        coeff.m_w_min << 0.0, -l_1, 0.0, 1.0;
        return;
    }

    // First basis vector that has no upward going wave amplitude
    coeff.m_w_min(0) = (b.x() - I * b.y()) * (l_1 - l_2) / 2.0 / b_mag;
    coeff.m_w_min(1) = b.z() * (l_2 - l_1) / 2.0 / b_mag - (l_1 + l_2) / 2.0;
    coeff.m_w_min(2) = 0.0;
    coeff.m_w_min(3) = 1.0;

    // Second basis vector that has no upward going wave amplitude
    coeff.m_w_plus(0) = -(l_1 + l_2) / 2.0 - b.z() / (l_1 + l_2);
    coeff.m_w_plus(1) = (b.x() + I * b.y()) * (l_1 - l_2) / 2.0 / b_mag;
    coeff.m_w_plus(2) = 1.0;
    coeff.m_w_plus(3) = 0.0;
}

void SpecularMagneticNewStrategy::propagateBackwards(std::vector<MatrixRTCoefficients_v3>& coeff,
                                                  const std::vector<Slice>& slices)
{
    const int size = static_cast<int>(coeff.size());
    for (int index = size - 2; index >= 0; --index) {
        const size_t i = static_cast<size_t>(index);
        const double t = slices[i].thickness();
        const auto kz = coeff[i].getKz();
        Eigen::Matrix4cd l = coeff[i].R1 * GetImExponential(kz(0) * t)
                             + coeff[i].T1 * GetImExponential(-kz(0) * t)
                             + coeff[i].R2 * GetImExponential(kz(1) * t)
                             + coeff[i].T2 * GetImExponential(-kz(1) * t);
        coeff[i].m_w_plus = l * coeff[i + 1].m_w_plus;
        coeff[i].m_w_min = l * coeff[i + 1].m_w_min;
    }
}

Eigen::Matrix2cd
SpecularMagneticNewStrategy::findNormalizationCoefficients(const MatrixRTCoefficients_v3& coeff)
{
    const Eigen::Vector2cd Ta = coeff.T1plus() + coeff.T2plus();
    const Eigen::Vector2cd Tb = coeff.T1min() + coeff.T2min();

    Eigen::Matrix2cd S;
    S << Ta(0), Tb(0), Ta(1), Tb(1);

    Eigen::Matrix2cd result;
    result << S(1, 1), -S(0, 1), -S(1, 0), S(0, 0);
    result /= S(0, 0) * S(1, 1) - S(1, 0) * S(0, 1);

    return result;
}

void SpecularMagneticNewStrategy::propagateForwards(std::vector<MatrixRTCoefficients_v3>& coeff,
                                                 const Eigen::Matrix2cd& weights)
{
    const complex_t a_plus = weights(0, 0);
    const complex_t b_plus = weights(1, 0);
    const complex_t a_min = weights(0, 1);
    const complex_t b_min = weights(1, 1);

    for (auto& term : coeff) {
        Eigen::Vector4cd w_plus = a_plus * term.m_w_plus + b_plus * term.m_w_min;
        Eigen::Vector4cd w_min = a_min * term.m_w_plus + b_min * term.m_w_min;
        term.m_w_plus = std::move(w_plus);
        term.m_w_min = std::move(w_min);
    }
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
    return {std::sqrt(a + 4. * M_PI * magnetic_SLD), std::sqrt(a - 4. * M_PI * magnetic_SLD)};
}

Eigen::Vector2cd checkForUnderflow(const Eigen::Vector2cd& eigenvs)
{
    auto lambda = [](complex_t value) { return std::abs(value) < 1e-40 ? 1e-40 : value; };
    return {lambda(eigenvs(0)), lambda(eigenvs(1))};
}

complex_t GetImExponential(complex_t exponent)
{
    if (exponent.imag() > -std::log(std::numeric_limits<double>::min()))
        return 0.0;
    return std::exp(I * exponent);
}
} // namespace
