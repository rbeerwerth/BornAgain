// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Models/ProjectionItems.cpp
//! @brief     Implements items related to projections over color map.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "ProjectionItems.h"

ProjectionContainerItem::ProjectionContainerItem() : SessionItem(Constants::ProjectionContainerType)
{
    const QString T_CHILDREN = "children tag";
    registerTag(T_CHILDREN, 0, -1,
                QStringList() << Constants::HorizontalLineMaskType
                              << Constants::VerticalLineMaskType);
    setDefaultTag(T_CHILDREN);
}
