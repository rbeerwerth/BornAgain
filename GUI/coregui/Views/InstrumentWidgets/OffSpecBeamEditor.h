// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/InstrumentWidgets/OffSpecBeamEditor.h
//! @brief     Defines class OffSpecBeamEditor
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef OFFSPECBEAMEDITOR_H
#define OFFSPECBEAMEDITOR_H

#include "SessionItemWidget.h"

class ComponentEditor;
class QGridLayout;
class OffSpecInstrumentItem;
class BeamItem;
class ColumnResizer;

//! GISAS beam editor. Operates on GISASInstrumentItem.

class BA_CORE_API_ OffSpecBeamEditor : public SessionItemWidget
{
    Q_OBJECT

public:
    explicit OffSpecBeamEditor(ColumnResizer* columnResizer, QWidget* parent = nullptr);

protected:
    void subscribeToItem();
    void unsubscribeFromItem();

private:
    OffSpecInstrumentItem* instrumentItem();
    BeamItem* beamItem();
    void onDialogRequest(SessionItem* item, const QString& name);

    ColumnResizer* m_columnResizer;
    ComponentEditor* m_intensityEditor;
    ComponentEditor* m_wavelengthEditor;
    ComponentEditor* m_inclinationEditor;
    ComponentEditor* m_azimuthalEditor;
    QGridLayout* m_gridLayout;
};

#endif // OFFSPECBEAMEDITOR_H
