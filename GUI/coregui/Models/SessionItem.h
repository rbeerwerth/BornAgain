// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Models/SessionItem.h
//! @brief     Defines class SessionItem
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2016
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   Céline Durniak, Marina Ganeva, David Li, Gennady Pospelov
//! @authors   Walter Van Herck, Joachim Wuttke
//
// ************************************************************************** //

#ifndef SESSIONITEM_H
#define SESSIONITEM_H

#include "ModelMapper.h"
#include "RealLimits.h"
#include "item_constants.h"
#include <QStringList>
#include <memory>

Q_DECLARE_METATYPE(RealLimits)

class SessionItemData;
class SessionItemTags;
class IPathTranslator;

class BA_CORE_API_ SessionItem
{
    friend class SessionModel;

public:
    static const QString P_NAME;

    explicit SessionItem(const QString& modelType = QString());
    virtual ~SessionItem();
    SessionModel* model() const;
    SessionItem* parent() const;

    // these functions work without tags and operate on all children
    QModelIndex index() const;
    bool hasChildren() const;
    int numberOfChildren() const;
    QVector<SessionItem*> children() const;
    SessionItem* childAt(int row) const;
    int rowOfChild(SessionItem* child) const;
    SessionItem* getChildOfType(const QString& type) const;
    QVector<SessionItem*> getChildrenOfType(const QString& model_type) const;
    SessionItem* takeRow(int row);

    // manage and check tags
    bool registerTag(const QString& name, int min = 0, int max = -1,
                     QStringList modelTypes = QStringList());
    bool isTag(const QString& name) const;
    SessionItemTags* sessionItemTags();
    QString tagFromItem(const SessionItem* item) const;
    bool acceptsAsDefaultItem(const QString& item_name) const;
    QVector<QString> acceptableDefaultItemTypes() const;

    // access tagged items
    SessionItem* getItem(const QString& tag = QString(), int row = 0) const;
    template <typename T> T& item(const QString& tag) const;
    QVector<SessionItem*> getItems(const QString& tag = QString()) const;
    bool insertItem(int row, SessionItem* item, const QString& tag = QString());
    SessionItem* takeItem(int row, const QString& tag);

    // convenience functions for properties
    SessionItem* addProperty(const QString& name, const QVariant& variant);
    QVariant getItemValue(const QString& tag) const;
    void setItemValue(const QString& tag, const QVariant& variant);

    // convenience functions for groups
    SessionItem* addGroupProperty(const QString& groupTag, const QString& groupType);
    SessionItem* setGroupProperty(const QString& groupTag, const QString& modelType) const;
    SessionItem* getGroupItem(const QString& groupName) const;
    template <typename T> T& groupItem(const QString& groupName) const;

    // access data stored in roles
    virtual QVariant data(int role) const;
    virtual bool setData(int role, const QVariant& value);
    QVector<int> getRoles() const;
    void emitDataChanged(int role = Qt::DisplayRole);

    // custom data types
    QString modelType() const;

    QVariant value() const;
    bool setValue(QVariant value);

    QString defaultTag() const;
    void setDefaultTag(const QString& tag);

    QString displayName() const;
    void setDisplayName(const QString& display_name);

    QString itemName() const;
    void setItemName(const QString& name);

    void setVisible(bool enabled);
    void setEnabled(bool enabled);
    void setEditable(bool enabled);
    bool isVisible() const;
    bool isEnabled() const;
    bool isEditable() const;

    RealLimits limits() const;
    SessionItem& setLimits(const RealLimits& value);

    int decimals() const;
    SessionItem& setDecimals(int n);

    QString toolTip() const;
    SessionItem& setToolTip(const QString& tooltip);

    QString editorType() const;
    SessionItem& setEditorType(const QString& editorType);

    ModelMapper* mapper();

    virtual QStringList translateList(const QStringList& list) const;
    void addTranslator(const IPathTranslator& translator);

private:
    void childDeleted(SessionItem* child);
    void setParentAndModel(SessionItem* parent, SessionModel* model);
    void setModel(SessionModel* model);
    int flags() const;
    void changeFlags(bool enabled, int flag);
    int getCopyNumberOfChild(const SessionItem* item) const;

    SessionItem* m_parent;
    SessionModel* m_model;
    QVector<SessionItem*> m_children;
    std::unique_ptr<SessionItemData> m_values;
    std::unique_ptr<SessionItemTags> m_tags;
    std::unique_ptr<ModelMapper> m_mapper;
    QVector<IPathTranslator*> m_translators;
};

template <typename T> T& SessionItem::item(const QString& tag) const
{
    T* t = dynamic_cast<T*>(getItem(tag));
    Q_ASSERT(t);
    return *t;
}

template <typename T> T& SessionItem::groupItem(const QString& groupName) const
{
    T* t = dynamic_cast<T*>(getGroupItem(groupName));
    Q_ASSERT(t);
    return *t;
}

#endif // SESSIONITEM_H
