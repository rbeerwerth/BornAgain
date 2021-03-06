// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Models/ComponentProxyModel.h
//! @brief     Defines class ComponentProxyModel
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef COMPONENTPROXYMODEL_H
#define COMPONENTPROXYMODEL_H

#include "ProxyModelStrategy.h"
#include "WinDllMacros.h"
#include <QAbstractProxyModel>
#include <QMap>
#include <QPersistentModelIndex>
#include <memory>

class SessionModel;
class ProxyModelStrategy;

//! Proxy model to adjust SessionModel for component editor (right bottom corner of SampleView
//! and similar).
//! The model hides all SessionItems which are not PropertyItems.
//! The model hides GroupPropertyItem children and shows grand-children of currently selected item
//! one level up.

class BA_CORE_API_ ComponentProxyModel : public QAbstractProxyModel
{
    Q_OBJECT

    friend class ProxyModelStrategy;

public:
    ComponentProxyModel(QObject* parent = nullptr);

    void setSessionModel(SessionModel* model);

    void setRootIndex(const QModelIndex& sourceRootIndex);

    void setProxyStrategy(ProxyModelStrategy* strategy);

    QModelIndex mapToSource(const QModelIndex& proxyIndex) const;
    QModelIndex mapFromSource(const QModelIndex& sourceIndex) const;

    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex& child) const;
    int rowCount(const QModelIndex& parent = QModelIndex()) const;
    int columnCount(const QModelIndex& parent = QModelIndex()) const;

    bool hasChildren(const QModelIndex& parent) const;

private slots:
    void sourceDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight,
                           const QVector<int>& roles = QVector<int>());
    void sourceRowsInserted(const QModelIndex& parent, int start, int end);
    void sourceRowsRemoved(const QModelIndex& parent, int start, int end);

private:
    void buildModelMap();
    void updateModelMap();

    SessionModel* m_model;
    std::unique_ptr<ProxyModelStrategy> m_proxyStrategy;
};

#endif // COMPONENTPROXYMODEL_H
