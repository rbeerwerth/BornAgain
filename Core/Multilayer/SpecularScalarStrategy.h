// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Multilayer/SpecularScalarStrategy.h
//! @brief     Defines class SpecularScalarStrategy.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef SPECULARSCALARSTRATEGY_H
#define SPECULARSCALARSTRATEGY_H

#include "ISpecularStrategy.h"
#include "ScalarRTCoefficients.h"
#include "Vectors3D.h"
#include <memory>
#include <vector>

class Slice;

//! Implements method 'execute' to compute refraction angles and transmission/reflection
//! coefficients for coherent wave propagation in a multilayer.
//! @ingroup algorithms_internal

class BA_CORE_API_ SpecularScalarStrategy : public ISpecularStrategy
{
public:
    //! Computes refraction angles and transmission/reflection coefficients
    //! for given coherent wave propagation in a multilayer.
    virtual ISpecularStrategy::coeffs_t Execute(const std::vector<Slice>& slices,
                                                const kvector_t& k) const override;

    virtual ISpecularStrategy::coeffs_t Execute(const std::vector<Slice>& slices,
                                                const std::vector<complex_t>& kz) const override;

private:
    virtual Eigen::Vector2cd transition(complex_t kzi, complex_t kzi1, double sigma,
                                        double thickness, const Eigen::Vector2cd& t_r1) const = 0;

    std::vector<ScalarRTCoefficients> computeTR(const std::vector<Slice>& slices,
                                                const std::vector<complex_t>& kz) const;

    static void setZeroBelow(std::vector<ScalarRTCoefficients>& coeff, size_t current_layer);

    bool calculateUpFromLayer(std::vector<ScalarRTCoefficients>& coeff,
                              const std::vector<Slice>& slices, const std::vector<complex_t>& kz,
                              size_t slice_index) const;
};

#endif // SPECULARSCALARSTRATEGY_H
