// ************************************************************************** //
//
//  Reflectometry simulation software prototype
//
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @authors   see AUTHORS
//
// ************************************************************************** //

#ifndef DAREFL_LAYEREDITOR_LAYEREDITORACTIONS_H
#define DAREFL_LAYEREDITOR_LAYEREDITORACTIONS_H

#include "darefl_export.h"
#include <QObject>
#include <memory>

namespace gui2 {

class SampleModel;
class LayerSelectionModel;

//! Handles user actions applied to layer tree.
//! Belongs to LayerEditor.

class DAREFLCORE_EXPORT LayerEditorActions : public QObject {
    Q_OBJECT

public:
    LayerEditorActions(QObject* parent = nullptr);
    ~LayerEditorActions();

    void setModel(SampleModel* model);

    void onAddLayer();
    void onAddMultiLayer();
    void onClone();
    void onRemove();
    void onMoveUp();
    void onMoveDown();

    void setSelectionModel(LayerSelectionModel* selection_model);

private:
    struct LayerEditorActionsImpl;
    std::unique_ptr<LayerEditorActionsImpl> p_impl;
};

} // namespace gui2

#endif // DAREFL_LAYEREDITOR_LAYEREDITORACTIONS_H
