// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Multilayer/SpecularStrategyBuilder.cpp
//! @brief     Implements class SpecularStrategyBuilder.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "SpecularStrategyBuilder.h"
#include "MultiLayerUtils.h"
#include "SpecularMagneticStrategy.h"
#include "SpecularScalarNCStrategy.h"
#include "SpecularScalarTanhStrategy.h"

std::unique_ptr<ISpecularStrategy> SpecularStrategyBuilder::build(const MultiLayer& sample,
                                                                  const bool magnetic)
{
    auto roughnessModel = sample.roughnessModel();

    if (magnetic) {
        if (MultiLayerUtils::hasRoughness(sample))
            throw std::runtime_error("Magnetic roughness not implemented");

        return std::make_unique<SpecularMagneticStrategy>();

    } else {
        if (roughnessModel == RoughnessModel::TANH || roughnessModel == RoughnessModel::DEFAULT) {
            return std::make_unique<SpecularScalarTanhStrategy>();

        } else if (roughnessModel == RoughnessModel::NEVOT_CROCE) {

            return std::make_unique<SpecularScalarNCStrategy>();

        } else
            throw std::logic_error("Invalid roughness model");
    }
}
