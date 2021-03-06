// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/MaskWidgets/RegionOfInterestView.h
//! @brief     Defines RegionOfInterestView class
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef REGIONOFINTERESTVIEW_H
#define REGIONOFINTERESTVIEW_H

#include "RectangleView.h"

//! The RegionOfInterest class represent view of RegionOfInterestItem on graphics scene.

class BA_CORE_API_ RegionOfInterestView : public RectangleView
{
    Q_OBJECT

public:
    int type() const { return MaskEditorHelper::REGIONOFINTEREST; }

protected:
    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*);
};

#endif // REGIONOFINTERESTVIEW_H
