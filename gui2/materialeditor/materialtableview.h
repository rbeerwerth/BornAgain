// ************************************************************************** //
//
//  Reflectometry simulation software prototype
//
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @authors   see AUTHORS
//
// ************************************************************************** //

#ifndef BORNAGAIN_GUI2_MATERIALEDITOR_MATERIALTABLEVIEW_H
#define BORNAGAIN_GUI2_MATERIALEDITOR_MATERIALTABLEVIEW_H

#include "darefl_export.h"
#include <QTableView>

namespace gui2 {

//! Extension of QTableView for material editing.
//! Provide better user experinece while navigating between cells.
//! Part of MaterialTableWidget.

class DAREFLCORE_EXPORT MaterialTableView : public QTableView {
public:
    using QTableView::QTableView;
    ~MaterialTableView() override;

    void setModel(QAbstractItemModel* model) override;

protected:
    void keyPressEvent(QKeyEvent* event) override;
    QModelIndex moveCursor(QAbstractItemView::CursorAction cursorAction,
                           Qt::KeyboardModifiers modifiers) override;

private:
    bool isTextField(const QModelIndex& index) const;
    bool isKeyboardEditable(const QModelIndex& index) const;
};

} // namespace gui2

#endif // BORNAGAIN_GUI2_MATERIALEDITOR_MATERIALTABLEVIEW_H