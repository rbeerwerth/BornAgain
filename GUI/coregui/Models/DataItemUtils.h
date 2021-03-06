// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Models/DataItemUtils.h
//! @brief     Defines namespace DataItemUtils
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef DATAITEMUTILS_H
#define DATAITEMUTILS_H

#include "WinDllMacros.h"

class IntensityDataItem;
class SessionItem;
class SpecularDataItem;

//! Utility functions for Intensity and Specular DataItems

namespace DataItemUtils
{
//! Returns IntensityDataItem contained as a child in givent parent.
BA_CORE_API_ IntensityDataItem* intensityDataItem(SessionItem* parent);

//! Returns SpecularDataItem contained as a child in givent parent.
BA_CORE_API_ SpecularDataItem* specularDataItem(SessionItem* parent);
} // namespace DataItemUtils

#endif //  DATAITEMUTILS_H
