// ************************************************************************** //
//
//  Reflectometry simulation software prototype
//
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @authors   see AUTHORS
//
// ************************************************************************** //

#ifndef DAREFL_LAYEREDITOR_LAYERTREEVIEW_H
#define DAREFL_LAYEREDITOR_LAYERTREEVIEW_H

#include "darefl_export.h"
#include <QTreeView>

namespace gui2 {

//! Extension of QTreeView for layer editing.

class DAREFLCORE_EXPORT LayerTreeView : public QTreeView {
public:
    using QTreeView::QTreeView;

    explicit LayerTreeView(QWidget* parent = nullptr);
    ~LayerTreeView() override;

    void setModel(QAbstractItemModel* model) override;
};

} // namespace gui2

#endif // DAREFL_LAYEREDITOR_LAYERTREEVIEW_H
