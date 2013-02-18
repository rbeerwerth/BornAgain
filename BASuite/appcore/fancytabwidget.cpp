#include "fancytabwidget.h"
#include "fancytabbar.h"
//#include "styledbar.h"
#include "stylehelper.h"

#include <QColorDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <QStackedLayout>
#include <QStatusBar>
#include <QPainter>
#include <QStackedWidget>


class FancyColorButton : public QWidget
{
public:
    FancyColorButton(QWidget *parent)
      : m_parent(parent)
    {
        setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);
    }

    void mousePressEvent(QMouseEvent *ev)
    {
        if (ev->modifiers() & Qt::ShiftModifier) {
            QColor color = QColorDialog::getColor(StyleHelper::requestedBaseColor(), m_parent);
            if (color.isValid())
                StyleHelper::setBaseColor(color);
        }
    }
private:
    QWidget *m_parent;
};


FancyTabWidget::FancyTabWidget(QWidget *parent)
    : QWidget(parent)
    , m_tabBar(0)
{

//    m_tabBar = new FancyTabBar(this);
    m_tabBar = new FancyTabBar2(this);

    m_selectionWidget = new QWidget(this);
    QVBoxLayout *selectionLayout = new QVBoxLayout;
    selectionLayout->setSpacing(0);
    selectionLayout->setMargin(0);

//    StyledBar *bar = new StyledBar(this);
//    QHBoxLayout *layout = new QHBoxLayout(bar);
//    layout->setMargin(0);
//    layout->setSpacing(10);
    //layout->addWidget(new FancyColorButton(this));

//    selectionLayout->addWidget(bar);
    selectionLayout->addWidget(m_tabBar);
    m_selectionWidget->setLayout(selectionLayout);
    m_selectionWidget->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

    m_cornerWidgetContainer = new QWidget(this);
    m_cornerWidgetContainer->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
    m_cornerWidgetContainer->setAutoFillBackground(true); //true

    QVBoxLayout *cornerWidgetLayout = new QVBoxLayout;
    cornerWidgetLayout->setSpacing(0);
    cornerWidgetLayout->setMargin(0);
    cornerWidgetLayout->addStretch();
    m_cornerWidgetContainer->setLayout(cornerWidgetLayout);

    selectionLayout->addWidget(m_cornerWidgetContainer, 0);

//    m_modesStack = new QStackedLayout;
    m_statusBar = new QStatusBar;
    m_statusBar->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    m_statusBar->showMessage("Hello world");
    //m_statusBar->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

    QVBoxLayout *vlayout = new QVBoxLayout;
    vlayout->setMargin(0);
    vlayout->setSpacing(0);
//    vlayout->addLayout(m_modesStack);
//    vlayout->addWidget(m_statusBar);


    m_stackedWidgets = new QStackedWidget;
//    m_stackedWidgets->addWidget(new ConfigurationPage);
//    m_stackedWidgets->addWidget(new UpdatePage);
//    m_stackedWidgets->addWidget(new QueryPage);

    vlayout->addWidget(m_stackedWidgets, 1);
    vlayout->addWidget(m_statusBar);

    QHBoxLayout *mainLayout = new QHBoxLayout;
    mainLayout->setMargin(0);
    mainLayout->setSpacing(1);
    mainLayout->addWidget(m_selectionWidget);
    mainLayout->addLayout(vlayout);
    setLayout(mainLayout);

    connect(m_tabBar, SIGNAL(currentChanged(int)), this, SLOT(showWidget(int)));

}


FancyTabWidget::~FancyTabWidget()
{
    delete m_tabBar;
}


void FancyTabWidget::insertTab(int index, QWidget *tab, const QIcon &icon, const QString &label)
{
    m_stackedWidgets->addWidget(tab);
    m_tabBar->insertTab(index, icon, label);
    m_tabBar->setTabEnabled(index, true);
}

//void FancyTabWidget::removeTab(int index)
//{
//    m_modesStack->removeWidget(m_modesStack->widget(index));
//    m_tabBar->removeTab(index);
//}


void FancyTabWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)
    QPainter painter(this);

    QRect rect = m_selectionWidget->rect().adjusted(0, 0, 1, 0);
    rect = style()->visualRect(layoutDirection(), geometry(), rect);
    StyleHelper::verticalGradient(&painter, rect, rect);
    painter.setPen(StyleHelper::borderColor());
    painter.drawLine(rect.topRight(), rect.bottomRight());

    QColor light = StyleHelper::sidebarHighlight();
    painter.setPen(light);
    painter.drawLine(rect.bottomLeft(), rect.bottomRight());
}


void FancyTabWidget::showWidget(int index)
{
    emit currentAboutToShow(index);
    m_stackedWidgets->setCurrentIndex(index);
    emit currentChanged(index);
}