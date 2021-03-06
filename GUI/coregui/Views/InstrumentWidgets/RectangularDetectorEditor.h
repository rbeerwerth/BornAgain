// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/InstrumentWidgets/RectangularDetectorEditor.h
//! @brief     Defines class RectangularDetectorEditor
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef RECTANGULARDETECTOREDITOR_H
#define RECTANGULARDETECTOREDITOR_H

#include "SessionItemWidget.h"
class RectangularDetectorItem;
class ComponentEditor;
class QGridLayout;

class BA_CORE_API_ RectangularDetectorEditor : public SessionItemWidget
{
    Q_OBJECT

public:
    RectangularDetectorEditor(QWidget* parent = nullptr);

public slots:
    void onPropertyChanged(const QString& propertyName);

protected:
    void subscribeToItem();
    void unsubscribeFromItem();
    RectangularDetectorItem* detectorItem();

private:
    void create_editors();
    void init_editors();
    void init_alignment_editors();

    ComponentEditor* m_xAxisEditor;
    ComponentEditor* m_yAxisEditor;
    ComponentEditor* m_resolutionFunctionEditor;
    ComponentEditor* m_alignmentEditor;
    ComponentEditor* m_positionsEditor;
    ComponentEditor* m_normalEditor;
    ComponentEditor* m_directionEditor;
    QGridLayout* m_gridLayout;
};

#endif // RECTANGULARDETECTOREDITOR_H
