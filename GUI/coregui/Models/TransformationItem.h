// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Models/TransformationItem.h
//! @brief     Defines class TransformationItem
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef TRANSFORMATIONITEM_H
#define TRANSFORMATIONITEM_H

#include "SessionGraphicsItem.h"

class BA_CORE_API_ TransformationItem : public SessionGraphicsItem
{
public:
    static const QString P_ROT;
    TransformationItem();
};

#endif // TRANSFORMATIONITEM_H
