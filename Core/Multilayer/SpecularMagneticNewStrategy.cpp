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
#include <iomanip>
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



std::pair<Eigen::Matrix2cd, Eigen::Matrix2cd> SpecularMagneticNewStrategy::computeDelta(MatrixRTCoefficients_v3& coeff, double thickness, double prefactor)
//Eigen::Matrix2cd SpecularMagneticNewStrategy::computeDelta(MatrixRTCoefficients_v3& coeff, double thickness, double prefactor)
{
    auto b = coeff.m_b;

//    std::cout << "b = " << b << std::endl;
    auto && cmpfct = [](const auto & cp1, const auto & cp2 ){ return std::norm(cp1) < std::norm(cp2); };

    Eigen::Matrix2cd result;
    Eigen::Matrix2cd deltaSmall;
    Eigen::Matrix2cd deltaLarge;
    auto Lp = prefactor * I * 0.5 * thickness * (coeff.m_lambda(0) + coeff.m_lambda(1));
    auto Lm = prefactor * I * 0.5 * thickness * (coeff.m_lambda(0) - coeff.m_lambda(1));

//    auto scaling1 = std::max( {std::exp(Lp), std::exp(-1. * Lp)}, cmpfct );
//    auto scaling2 = std::max( {std::exp(Lm), std::exp(-1. * Lm)}, cmpfct );
//    auto scaling1 = 1.;
//    auto scaling2 = 1.;

//    Lp *= prefactor;
//    Lm *= prefactor;

//    std::cout << "======================= computeDelta(" << prefactor << ")" << std::endl;


//    auto Q = Eigen::Matrix2cd::Zero();
    Eigen::Matrix2cd Q;

    auto factor1 = std::sqrt(2. * ( 1. + b.z()));
    auto factor2 = std::sqrt(2. * ( 1. - b.z()));

    Q << (1. + b.z()) / factor1, (b.z() - 1.) / factor2,
            (b.x() + I * b.y()) / factor1, (b.x() + I * b.y()) / factor2;

//    std::cout << "Q.+ * Q = " << Q.adjoint() * Q << std::endl;
//    Eigen::Matrix2cd exp2;
//    return result;
//    auto && maxExp = std::max( {std::exp(Lp), std::exp(Lm), std::exp(-Lm)}, cmpfct );


//    auto scaling1 = 2.;
//    auto scaling2 = 2.;

//    std::cout << "scaling1 = " << scaling1 << std::endl;
//    std::cout << "scaling2 = " << scaling2 << std::endl;

    auto exp1 = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({std::exp(Lp), std::exp(Lp) }) ) ;
//    exp1 /= scaling1;
    auto exp2 = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({std::exp(Lm), std::exp(-Lm)}) );
//    exp2 /= scaling2;



//    Eigen::Matrix2cd result2

//    result = exp2 * Q * exp1 * Q.adjoint();
//    result = Q * Q.adjoint();
    if ( std::abs(b.mag() - 1.) < std::numeric_limits<double>::epsilon() * 10.)
        result = exp1 * Q * exp2 * Q.adjoint();
    else if(b.mag() == 0.)
        result = Eigen::Matrix2cd(exp1) * Eigen::Matrix2cd(exp2);
    else
        throw std::runtime_error("Broken magnetic field vector");

//    std::cout << "Q = " << Q << std::endl;
//    std::cout << "exp1 = " << exp1 << std::endl;
//    std::cout << "exp2 = " << exp2 << std::endl;
//    std::cout << "delta = " << result << std::endl;

    auto det = std::exp(Lp) * std::exp(Lp) * std::exp(Lm) * std::exp(-Lm);
    auto det2 = result(0, 0) * result(1, 1) - result(0, 1) * result(1, 0);



    // separate matrix into medium and small part
    auto exp2Large = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({std::exp(Lm), complex_t(0., 0.)}) );
    auto exp2Small = Eigen::Matrix2cd( Eigen::DiagonalMatrix<complex_t, 2>({complex_t(0., 0.), std::exp(-Lm)}) );
    if(std::norm(std::exp(-Lm)) > std::norm(std::exp(Lm)) )
        std::swap(exp2Large, exp2Small);

//    result = Q * exp2 * Q.adjoint();
    if (std::abs(b.mag() - 1.) < std::numeric_limits<double>::epsilon() * 10.)
    {
        deltaSmall = exp1 * Q * exp2Small * Q.adjoint();
        deltaLarge = exp1 * Q * exp2Large * Q.adjoint();
    }else if(b.mag() == 0.)
    {
        deltaSmall = exp1 * (exp2Small + exp2Large);
        deltaLarge = Eigen::Matrix2cd::Zero() ;//exp1 * exp2Large;
    }
    else
        throw std::runtime_error("Broken magnetic field vector");

//    std::cout << "result = " << result << std::endl;
//    std::cout << "deltaS = " << deltaSmall << std::endl;
//    std::cout << "deltaL = " << deltaLarge << std::endl;
//    std::cout << "deltaS + deltaL" << deltaSmall + deltaLarge << std::endl;

    auto && fancyDet = [](const Eigen::Matrix2cd matL, const Eigen::Matrix2cd matS){
        auto det = matL(0, 0) * matL(1, 1) - matL(0, 1) * matL(1, 0) + \
                matL(0, 0) * matS(1, 1) + matL(1, 1) * matS(0, 0) - matL(0, 1) * matS(1, 0) - matL(1, 0) * matS(0, 1) + \
                matS(0, 0) * matS(1, 1) - matS(0, 1) * matS(1, 0);
        return det;
    };

    auto separatedDet = fancyDet(deltaLarge, deltaSmall);

//    std::cout << "det1(Delta) = " << det << std::endl;
//    std::cout << "det2(Delta) = " << det2 << std::endl;
//    std::cout << "fancydet(Delta) = " << separatedDet << std::endl;

//    return result;
    return std::make_pair(deltaLarge, deltaSmall);

    // lazy way
//    auto pm = prefactor * I * thickness * computeP(coeff);
//    auto expmatrix = pm.exp();
//    return expmatrix;
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

//    auto testindex{1};


//    if(testindex >= slices.size())
//        throw  std::runtime_error("Trying to access invalid slice");


//    auto pm0  = computeP(result[testindex]);
//    auto pmi0 = computeInverseP(result[testindex]);

//    std::cout << "test" << std::endl;
//    std::cout << "pm0 = " << pm0 << std::endl;
//    std::cout << "pmi0 = " << pmi0 << std::endl;


//    std::cout << "pm0 * pmi0 = " << pm0 * pmi0 << std::endl;


    // calculate the matrices M_i
    for (size_t i = 0, interfaces = slices.size() - 1; i < interfaces; ++i) {

//        std::cout << "===================================================================================================\ni = "<< i << std::endl;
//        auto bi = result[i].m_b;

//        std::cout << "b = " << bi << std::endl;

//        std::cout << "lp = " << result[i].m_lambda(0) << " lm = " << result[i].m_lambda(1) << std::endl;
//        std::cout << "Lp = " << result[i].m_lambda(0) + result[i].m_lambda(1) << " Lm = " << result[i].m_lambda(0) - result[i].m_lambda(1) << std::endl;
//        std::cout << "exponentials: " << std::exp(I * slices[i].thickness() * result[i].m_lambda(0) ) << " " <<
//                                         std::exp(-I * slices[i].thickness() * result[i].m_lambda(0) ) << " " <<
//                                         std::exp(I * slices[i].thickness() * result[i].m_lambda(1) ) << " " <<
//                                         std::exp(-I * slices[i].thickness() * result[i].m_lambda(1) ) << std::endl;

        auto mproduct = Eigen::Matrix2cd( computeInverseP(result[i]) * computeP(result[i+1]) );
        auto mp = Eigen::Matrix2cd::Identity() + mproduct;
        auto mm = Eigen::Matrix2cd::Identity() - mproduct;
//        auto pm1 = ;
        auto deltaTemp = computeDelta(result[i], slices[i].thickness(), 1.);
        auto deltaInv = computeDelta(result[i], slices[i].thickness(), -1.);

        auto delta = std::get<0>(deltaTemp) + std::get<1>(deltaTemp);
//        auto delta = Eigen::Matrix2cd( std::get<0>(deltaTemp) + std::get<1>(deltaTemp) );
//        auto deltaInv = Eigen::Matrix2cd( std::get<0>(deltaInvTemp) + std::get<1>(deltaInvTemp) );

//        auto delta    = computeDelta(result[i], slices[i].thickness(), 1.);
//        auto deltaInv = computeDelta(result[i], slices[i].thickness(), -1.);


//        auto Mi = Eigen::Matrix4cd::Zero();
//        std::cout << "delta = " << delta << std::endl;
//        std::cout << "delta^* = " << deltaInv << std::endl;

//        auto detDeltaInv = deltaInv(0, 0) * deltaInv(1, 1) - deltaInv(1, 0) * deltaInv(0, 1);

//        auto Lp = -1.* I * 0.5 * slices[i].thickness() * (result[i].m_lambda(0) + result[i].m_lambda(1));
//        auto Lm = -1.* I * 0.5 * slices[i].thickness() * (result[i].m_lambda(0) - result[i].m_lambda(1));

//        auto detExact = std::exp(Lp) * std::exp(Lp) * std::exp(Lm) * std::exp(-Lm);

//        std::cout << "det(delta) = " << delta(0, 0) * delta(1, 1) - delta(1, 0) * delta(0, 1) << std::endl;
//        std::cout << "det(delta^*) = " << detDeltaInv << std::endl;


        result[i].MiL = Eigen::Matrix4cd::Zero();
        result[i].MiL.block<2,2>(0, 0) = std::get<0>(deltaInv) * mp;
        result[i].MiL.block<2,2>(0, 2) = std::get<0>(deltaInv) * mm;
//        result[i].MiL.block<2,2>(2, 0) = delta * mm;
//        result[i].MiL.block<2,2>(2, 2) = delta * mp;

        result[i].MiS = Eigen::Matrix4cd::Zero();
        result[i].MiS.block<2,2>(0, 0) = std::get<1>(deltaInv) * mp;
        result[i].MiS.block<2,2>(0, 2) = std::get<1>(deltaInv) * mm;
        result[i].MiS.block<2,2>(2, 0) = delta * mm;
        result[i].MiS.block<2,2>(2, 2) = delta * mp;

//        result[i].Mi.block<2,2>(0, 0) = mp;
//        result[i].Mi.block<2,2>(0, 2) = mm;
//        result[i].Mi.block<2,2>(2, 0) = mm;
//        result[i].Mi.block<2,2>(2, 2) = mp;


        result[i].MiL /= 0.5;
        result[i].MiS /= 0.5;
//        result[i].Mi /= detExact;

//        std::cout << "MiL = " << result[i].MiL << std::endl;
//        std::cout << "MiS = " << result[i].MiS << std::endl;
//        std::cout << "det(Mi) = " << result[i].Mi(0, 1) * result[i].Mi(1, 0) - result[i].Mi(1, 1) * result[i].Mi(0, 0) << std::endl;


//        std::cout << "==============================================" << std::endl;
//        std::cout << std::setprecision(16) << "pm-1 pm+1 = " << mproduct << std::endl;
//        std::cout << std::setprecision(16) << "delta*L = " << std::get<0>(deltaInv) << std::endl;
//        std::cout << std::setprecision(16) << "delta*S = " << std::get<1>(deltaInv) << std::endl;
//        std::cout << std::setprecision(16) << "delta  = " << delta << std::endl;
//        std::cout << "==============================================" << std::endl;




    }


    // calculate the total tranfer matrix M

    if(slices.size() == 2)
    {
//        result[0].ML = Eigen::Matrix4cd::Zero();
        result[0].MM = result[0].MiL;
        result[0].MS = result[0].MiS;
    }
    else
    {
//        result[slices.size()-2].ML = Eigen::Matrix4cd::Zero();
        result[slices.size()-2].MM = result[slices.size()-2].MiL;
        result[slices.size()-2].MS = result[slices.size()-2].MiS;
                ;
    }
    for (int i = slices.size() - 3; i >= 0; --i)
    {
//        result[i].ML = result[i].MiL * result[i+1].ML + result[i].MiS * result[i+1].ML + result[i].MiL * result[i+1].MM;
//        result[i].MM = result[i].MiS * result[i+1].MM + result[i].MiL * result[i+1].MS;
//        result[i].ML = Eigen::Matrix4cd::Zero();
//        result[i].MM = result[i].MiS * result[i+1].ML + result[i].MiL * result[i+1].MM;
        result[i].MM = result[i].MiL * result[i+1].MM + result[i].MiS * result[i+1].MM + result[i].MiL * result[i+1].MS;
        result[i].MS = result[i].MiS * result[i+1].MS;
    }


//    std::cout << "ML = " << result.front().getML() << std::endl;
//    std::cout << "MM = " << result.front().getMM() << std::endl;
//    std::cout << "MS = " << result.front().getMS() << std::endl;

    // extract R

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
