// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/StandardSamples/ThickAbsorptiveSampleBuilder.h
//! @brief     Defines class to build thick highly-absorptive sample with roughness
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef THICKABSORPTIVESAMPLEBUILDER_H
#define THICKABSORPTIVESAMPLEBUILDER_H

#include "IMultiLayerBuilder.h"

class BA_CORE_API_ ThickAbsorptiveSampleBuilder : public IMultiLayerBuilder
{
public:
    ThickAbsorptiveSampleBuilder();
    MultiLayer* buildSample() const override;
};

#endif // THICKABSORPTIVESAMPLEBUILDER_H
