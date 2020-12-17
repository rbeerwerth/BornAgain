// ************************************************************************** //
//
//  Reflectometry simulation software prototype
//
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @authors   see AUTHORS
//
// ************************************************************************** //

#ifndef BORNAGAIN_GUI2_MAINWINDOW_FANCYTAB_H
#define BORNAGAIN_GUI2_MAINWINDOW_FANCYTAB_H

#include "darefl_export.h"
#include <QColor>
#include <QWidget>

class QLabel;
class QString;

namespace gui2 {

class DAREFLCORE_EXPORT FancyTab : public QWidget {
    Q_OBJECT

public:
    FancyTab(const QString& title, QWidget* parent = nullptr);

    void setSelected(bool value);

signals:
    void clicked();

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent* event) override;
    void enterEvent(QEvent*) override;
    void leaveEvent(QEvent*) override;

private:
    QLabel* m_label{nullptr};
    bool m_isSelected{false};
    QColor m_widgetColor;
};

} // namespace gui2

#endif // BORNAGAIN_GUI2_MAINWINDOW_FANCYTAB_H