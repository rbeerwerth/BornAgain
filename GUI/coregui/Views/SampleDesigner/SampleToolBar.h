// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/SampleDesigner/SampleToolBar.h
//! @brief     Defines class SampleToolBar
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef SAMPLETOOLBAR_H
#define SAMPLETOOLBAR_H

#include "StyledToolBar.h"
#include <QTreeView>

class QAction;
class QToolButton;
class QToolBar;
class QComboBox;
class QString;
class QButtonGroup;
class SampleModel;
class SampleViewActions;

//! The SampleToolBar class represents a main toolbar on top of SampleView window

class BA_CORE_API_ SampleToolBar : public StyledToolBar
{
    Q_OBJECT

public:
    explicit SampleToolBar(SampleViewActions* sampleActions = nullptr, QWidget* parent = nullptr);

signals:
    void deleteItems();
    void selectionMode(int);
    void centerView();
    void smartAlign();
    void changeScale(double);

public slots:
    void onViewSelectionMode(int);
    void onScaleComboChanged(const QString&);
    void onMaterialEditorCall();

private:
    QButtonGroup* m_pointerModeGroup;
    QToolButton* m_removeButton;
    QToolButton* m_centerViewButton;
    QComboBox* m_scaleCombo;
    QToolButton* m_materialEditorButton;
    QToolButton* m_RealSpaceViewerButton;
    SampleViewActions* m_sampleViewActions;
};

#endif // SAMPLETOOLBAR_H
