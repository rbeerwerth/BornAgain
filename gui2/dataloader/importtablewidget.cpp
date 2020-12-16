// ************************************************************************** //
//
//  Reflectometry simulation software prototype
//
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @authors   see AUTHORS
//
// ************************************************************************** //

#include "gui2/dataloader/importtablewidget.h"
#include "gui2/dataloader/dataloader_types.h"
#include "gui2/dataloader/importtablemodel.h"
#include "mvvm/viewmodel/viewmodeldelegate.h"
#include <QStandardItemModel>
#include <QTableView>
#include <QVBoxLayout>

namespace DaRefl {

ImportTableWidget::ImportTableWidget(QWidget* parent)
    : QWidget(parent)
    , m_tableModel(new ImportTableModel(this))
    , m_tableView(new QTableView)
    , m_delegate(new ModelView::ViewModelDelegate) {
    auto layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_tableView);

    m_tableView->setItemDelegate(m_delegate.get());
    m_tableView->setModel(m_tableModel);
}

ImportTableWidget::~ImportTableWidget() = default;

void ImportTableWidget::setRawData(const std::vector<std::vector<std::string>>& table_data) {
    m_tableModel->setRawData(table_data);
}

std::vector<ColumnInfo> ImportTableWidget::columnInfo() const {
    return m_tableModel->columnInfo();
}

} // namespace DaRefl
