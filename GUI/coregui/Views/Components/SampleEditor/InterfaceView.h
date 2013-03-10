#ifndef INTERFACEVIEW_H
#define INTERFACEVIEW_H


#include <QGraphicsWidget>
#include <QPainter>

//- -------------------------------------------------------------------
//! @class InterfaceView
//! @brief Graphics representation of Particle in SampleEditorScene
//- -------------------------------------------------------------------
class InterfaceView : public QGraphicsWidget
{
public:
    InterfaceView(const QColor &color, const QColor &textColor, const QString &caption, QGraphicsItem *parent = 0)
        : QGraphicsWidget(parent)
        , caption(caption)
        , color(color)
        , textColor(textColor)
    {
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget * = 0)
    {
        QFont font;
        font.setPixelSize(0.75 * qMin(boundingRect().width(), boundingRect().height()));

        painter->fillRect(boundingRect(), color);
        painter->save();
        painter->setFont(font);
        painter->setPen(textColor);
        painter->drawText(boundingRect(), Qt::AlignCenter, caption);
        painter->restore();
    }

private:
    QString caption;
    QColor color;
    QColor textColor;
};

#endif // INTERFACEVIEW_H
