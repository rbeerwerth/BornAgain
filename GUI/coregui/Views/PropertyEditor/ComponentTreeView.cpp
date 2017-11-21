// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/PropertyEditor/ComponentTreeView.cpp
//! @brief     Implements class ComponentTreeView
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2016
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   Céline Durniak, Marina Ganeva, David Li, Gennady Pospelov
//! @authors   Walter Van Herck, Joachim Wuttke
//
// ************************************************************************** //

#include "ComponentTreeView.h"
#include "ComponentProxyModel.h"
#include "StyleUtils.h"
#include "SessionModelDelegate.h"
#include "SessionModel.h"
#include <QTreeView>
#include <QBoxLayout>
#include <QStandardItemModel>

ComponentTreeView::ComponentTreeView(QWidget* parent)
    : QWidget(parent)
    , m_tree(new QTreeView)
    , m_delegate(new SessionModelDelegate(this))
    , m_proxyModel(new ComponentProxyModel(this))
    , m_placeHolderModel(new QStandardItemModel)
{
    auto layout = new QVBoxLayout;

    layout->setMargin(0);
    layout->setSpacing(0);
    layout->addWidget(m_tree);

    setLayout(layout);

    QStringList labels = {"Name", "Value"};
    m_placeHolderModel->setHorizontalHeaderLabels(labels);

    StyleUtils::setPropertyStyle(m_tree);
    m_tree->setRootIsDecorated(false);
    m_tree->setModel(m_placeHolderModel);
    m_tree->setItemDelegate(m_delegate);
    m_tree->setEditTriggers(QAbstractItemView::AllEditTriggers);
}

void ComponentTreeView::setItem(SessionItem* item)
{
    if (!item) {
        setModel(nullptr);
        return;
    }
    setModel(item->model());
    setRootIndex(item->index());
    m_tree->expandAll();
}

void ComponentTreeView::setModel(SessionModel* model)
{
    m_proxyModel->setSessionModel(model);
    if (model)
        m_tree->setModel(m_proxyModel);
    else
        m_tree->setModel(m_placeHolderModel);

}

void ComponentTreeView::setRootIndex(const QModelIndex& index)
{
    if (QWidget* editor = m_tree->indexWidget(m_tree->currentIndex()))
        m_delegate->closeEditor(editor, QAbstractItemDelegate::NoHint);
    Q_ASSERT(m_proxyModel);
    m_proxyModel->setRootIndex(index);
}

QTreeView* ComponentTreeView::treeView()
{
    return m_tree;
}

