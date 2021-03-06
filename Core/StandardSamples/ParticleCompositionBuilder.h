// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/StandardSamples/ParticleCompositionBuilder.h
//! @brief     Defines class ParticleCompositionBuilder.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef PARTICLECOMPOSITIONBUILDER_H
#define PARTICLECOMPOSITIONBUILDER_H

#include "IMultiLayerBuilder.h"

//! Builds sample: two layers of spheres at hex lattice.
//! @ingroup standard_samples

class BA_CORE_API_ ParticleCompositionBuilder : public IMultiLayerBuilder
{
public:
    ParticleCompositionBuilder() {}
    MultiLayer* buildSample() const;
};

#endif // PARTICLECOMPOSITIONBUILDER_H
