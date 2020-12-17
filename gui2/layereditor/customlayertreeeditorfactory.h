// ************************************************************************** //
//
//  Reflectometry simulation software prototype
//
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @authors   see AUTHORS
//
// ************************************************************************** //

#ifndef BORNAGAIN_GUI2_LAYEREDITOR_CUSTOMLAYERTREEEDITORFACTORY_H
#define BORNAGAIN_GUI2_LAYEREDITOR_CUSTOMLAYERTREEEDITORFACTORY_H

#include "darefl_export.h"
#include "mvvm/editors/defaulteditorfactory.h"

namespace gui2 {

class ApplicationModels;

//! Custom editor factory for LayerTreeView. Substitutes default ExternalProperty editor
//! with custom one, which will offer the choice between all defined materials.

class DAREFLCORE_EXPORT CustomLayerTreeEditorFactory : public ModelView::DefaultEditorFactory {
public:
    CustomLayerTreeEditorFactory(ApplicationModels* models);
    ~CustomLayerTreeEditorFactory();

    std::unique_ptr<ModelView::CustomEditor> createEditor(const QModelIndex& index) const;

private:
    ApplicationModels* m_models;
};

} // namespace gui2

#endif // BORNAGAIN_GUI2_LAYEREDITOR_CUSTOMLAYERTREEEDITORFACTORY_H