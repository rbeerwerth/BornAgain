// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/StandardSamples/MultiLayerWithNCRoughnessBuilder.h
//! @brief     Defines class MultiLayerWithNCRoughnessBuilder.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef MULTILAYERWITHNCROUGHNESSBUILDER_H
#define MULTILAYERWITHNCROUGHNESSBUILDER_H

#include "IMultiLayerBuilder.h"
#include "MultiLayerWithRoughnessBuilder.h"

class ISample;

//! Builds sample: layers with correlated roughness.
//! @ingroup standard_samples

class BA_CORE_API_ MultiLayerWithNCRoughnessBuilder : public MultiLayerWithRoughnessBuilder
{
public:
    MultiLayerWithNCRoughnessBuilder();
    MultiLayer* buildSample() const override;

private:
};

#endif // MULTILAYERWITHNCROUGHNESSBUILDER_H
