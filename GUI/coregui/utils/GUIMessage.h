// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/utils/GUIMessage.h
//! @brief     Defines class GUIMessage.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef GUIMESSAGE_H
#define GUIMESSAGE_H

#include "WinDllMacros.h"
#include <QString>

class QObject;

class BA_CORE_API_ GUIMessage
{
public:
    GUIMessage(const QString& senderName, const QString& messageType,
               const QString& messageDescription);
    GUIMessage(const QObject* sender, const QString& messageType,
               const QString& messageDescription);

    QString senderName() const;
    QString messageType() const;
    QString messageDescription() const;

    QString text() const;

    const QObject* sender() const;

private:
    const QObject* m_sender;
    QString m_senderName;
    QString m_messageType;
    QString m_messageDescription;
};

#endif // GUIMESSAGE_H
