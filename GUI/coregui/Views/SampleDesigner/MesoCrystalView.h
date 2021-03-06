// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/SampleDesigner/MesoCrystalView.h
//! @brief     Defines class MesoCrystalView
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef MESOCRYSTALVIEW_H
#define MESOCRYSTALVIEW_H

#include "ConnectableView.h"

//! Class representing view of a meso crystal item
class BA_CORE_API_ MesoCrystalView : public ConnectableView
{
    Q_OBJECT

public:
    MesoCrystalView(QGraphicsItem* parent = 0);

    int type() const { return ViewTypes::PARTICLE; }

    void addView(IView* childView, int row = 0);
};

#endif // MESOCRYSTALVIEW_H
