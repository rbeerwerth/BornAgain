// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/StandardSamples/CylindersAndPrismsBuilder.h
//! @brief     Defines class CylindersAndPrismsBuilder.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef CYLINDERSANDPRISMSBUILDER_H
#define CYLINDERSANDPRISMSBUILDER_H

#include "IMultiLayerBuilder.h"

class ISample;

//! Builds sample: mixture of cylinders and prisms without interference (IsGISAXS example #1).
//! @ingroup standard_samples

class BA_CORE_API_ CylindersAndPrismsBuilder : public IMultiLayerBuilder
{
public:
    CylindersAndPrismsBuilder();
    MultiLayer* buildSample() const;
};

#endif // CYLINDERSANDPRISMSBUILDER_H
