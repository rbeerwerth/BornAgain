// ************************************************************************** //
//
//  Reflectometry simulation software prototype
//
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @authors   see AUTHORS
//
// ************************************************************************** //

#ifndef DAREFL_DATALOADER_LOADERPREVIEWPANEL_H
#define DAREFL_DATALOADER_LOADERPREVIEWPANEL_H

#include "darefl_export.h"
#include <QWidget>

class QTabWidget;

namespace gui2 {

class ImportTextView;
class ImportTableWidget;

class ParserInterface;
struct ColumnInfo;

//! Panel with settings for DataLoaderDialog.
//! Located on its right side, contains text and table views.

class DAREFLCORE_EXPORT LoaderPreviewPanel : public QWidget {
    Q_OBJECT

public:
    LoaderPreviewPanel(QWidget* parent = nullptr);

    void showData(const ParserInterface* parser);

    std::vector<ColumnInfo> columnInfo() const;

    void clearPanel();

private:
    ImportTextView* m_textView{nullptr};
    ImportTableWidget* m_tableWidget{nullptr};
    QTabWidget* m_tabWidget{nullptr};
};

} // namespace gui2

#endif // DAREFL_DATALOADER_LOADERPREVIEWPANEL_H
