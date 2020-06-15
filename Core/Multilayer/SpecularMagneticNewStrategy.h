// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Multilayer/SpecularMagneticStrategy.h
//! @brief     Defines class SpecularMagneticNewStrategy.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2020
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef SPECULARMAGNETICNEWSTRATEGY_H
#define SPECULARMAGNETICNEWSTRATEGY_H

#include "ISpecularStrategy.h"
#include "MatrixRTCoefficients_v3.h"
#include "Vectors3D.h"
#include <memory>
#include <vector>

class Slice;

//! Implements the matrix formalism for the calculation of wave amplitudes of
//! the coherent wave solution in a multilayer with magnetization.
//! @ingroup algorithms_internal
class BA_CORE_API_ SpecularMagneticNewStrategy : public ISpecularStrategy
{
public:
    //! Computes refraction angle reflection/transmission coefficients
    //! for given sliced multilayer and wavevector k
    ISpecularStrategy::coeffs_t Execute(const std::vector<Slice>& slices, const kvector_t& k) const;

    //! Computes refraction angle reflection/transmission coefficients
    //! for given sliced multilayer and a set of kz projections corresponding to each slice
    ISpecularStrategy::coeffs_t Execute(const std::vector<Slice>& slices,
                                        const std::vector<complex_t>& kz) const;

    static std::vector<MatrixRTCoefficients_v3> computeTR(const std::vector<Slice>& slices,
                                                          const std::vector<complex_t>& kzs);

//    static std::pair<Eigen::Matrix2cd, Eigen::Matrix2cd> computeDelta(
//                    MatrixRTCoefficients_v3& coeff, double thickness, double prefactor);

    // old things
    //! Computes frobenius matrices for multilayer solution
    static void calculateTR(MatrixRTCoefficients_v3& coeff);
    static void calculateZeroFieldTR(MatrixRTCoefficients_v3& coeff);

    static void setNoTransmission(MatrixRTCoefficients_v3& coeff);

    //! initializes reflectionless bottom boundary condition.
    static void nullifyBottomReflection(MatrixRTCoefficients_v3& coeff);

    //! Propagates boundary conditions from the bottom to the top of the layer stack.
    //! Used to compute boundary conditions from the bottom one (with nullified reflection)
    static void propagateBackwards(std::vector<MatrixRTCoefficients_v3>& coeff,
                                   const std::vector<Slice>& slices);

    //! finds linear coefficients for normalizing transmitted wave to unity.
    //! The left column of the returned matrix corresponds to the coefficients for pure spin-up
    //! wave, while the right column - to the coefficients for the spin-down one.
    static Eigen::Matrix2cd findNormalizationCoefficients(const MatrixRTCoefficients_v3& coeff);

    //! makes a linear combination of boundary conditions with using the given weights for each
    //! coefficient in the vector.
    static void propagateForwards(std::vector<MatrixRTCoefficients_v3>& coeff,
                                  const Eigen::Matrix2cd& weights);
};

#endif // SPECULARMAGNETICSTRATEGY_H
