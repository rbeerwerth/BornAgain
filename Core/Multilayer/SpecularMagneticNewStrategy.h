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

#include "Core/Multilayer/ISpecularStrategy.h"
#include "Core/Multilayer/MatrixRTCoefficients_v3.h"
#include "Core/Vector/Vectors3D.h"
#include <memory>
#include <vector>

class Slice;

//! Implements the magnetic Fresnel computation with Nevot-Croce roughness
//!
//! Implements the transfer matrix formalism for the calculation of wave
//! amplitudes of the coherent wave solution in a multilayer with magnetization.
//! For a description, see internal
//! document "Polarized Implementation of the Transfer Matrix Method"
//!
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

    static void computeInterfaceTransferMatrices(std::vector<MatrixRTCoefficients_v3>& coeff,
                                                 const std::vector<Slice>& slices);
    static void computeTotalTransferMatrices(std::vector<MatrixRTCoefficients_v3>& coeff);
    static void calculateAmplitudes(std::vector<MatrixRTCoefficients_v3>& coeff,
                                    const std::vector<Slice>& slices);
};

#endif // SPECULARMAGNETICSTRATEGY_H
