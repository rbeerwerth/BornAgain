// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/mainwindow/aboutapplicationdialog.h
//! @brief     Defines class AboutApplicationDialog
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef ABOUTAPPLICATIONDIALOG_H
#define ABOUTAPPLICATIONDIALOG_H

#include "WinDllMacros.h"
#include <QDialog>

class QBoxLayout;

//! About BornAgain dialog.

class BA_CORE_API_ AboutApplicationDialog : public QDialog
{
    Q_OBJECT
public:
    AboutApplicationDialog(QWidget* parent = 0);

private:
    QBoxLayout* createLogoLayout();
    QBoxLayout* createTextLayout();
    QBoxLayout* createButtonLayout();
};

#endif // ABOUTAPPLICATIONDIALOG_H
