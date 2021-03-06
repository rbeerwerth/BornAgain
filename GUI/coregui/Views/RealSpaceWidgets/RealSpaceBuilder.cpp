// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/RealSpaceWidgets/RealSpaceBuilder.cpp
//! @brief     Implements class RealSpaceBuilder
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "RealSpaceBuilder.h"
#include "ExternalProperty.h"
#include "InterferenceFunctionItems.h"
#include "InterferenceFunctions.h"
#include "Lattice2DItems.h"
#include "LayerItem.h"
#include "MesoCrystalItem.h"
#include "MultiLayerItem.h"
#include "Particle.h"
#include "Particle3DContainer.h"
#include "ParticleCompositionItem.h"
#include "ParticleCoreShell.h"
#include "ParticleCoreShellItem.h"
#include "ParticleDistributionItem.h"
#include "ParticleItem.h"
#include "ParticleLayoutItem.h"
#include "RealSpaceBuilderUtils.h"
#include "RealSpaceCanvas.h"
#include "RealSpaceModel.h"
#include "RealSpacePositionBuilder.h"
#include "SessionItem.h"
#include "TransformTo3D.h"
#include "Units.h"
#include "VectorItem.h"
#include <QDebug>
#include <ba3d/model/layer.h>

namespace
{
std::unique_ptr<IInterferenceFunction> GetInterferenceFunction(const SessionItem& layoutItem);
}

RealSpaceBuilder::RealSpaceBuilder(QWidget* parent) : QWidget(parent) {}

RealSpaceBuilder::~RealSpaceBuilder() {}

void RealSpaceBuilder::populate(RealSpaceModel* model, const SessionItem& item,
                                const SceneGeometry& sceneGeometry,
                                const RealSpace::Camera::Position& cameraPosition)
{
    // default value of cameraPosition is in RealSpaceBuilder.h

    model->defCamPos = cameraPosition;

    if (item.modelType() == Constants::MultiLayerType)
        populateMultiLayer(model, item, sceneGeometry);

    else if (item.modelType() == Constants::LayerType)
        populateLayer(model, item, sceneGeometry);

    else if (item.modelType() == Constants::ParticleLayoutType)
        populateLayout(model, item, sceneGeometry);

    else if (item.modelType() == Constants::ParticleType)
        populateParticleFromParticleItem(model, item);

    else if (item.modelType() == Constants::ParticleCompositionType)
        populateParticleFromParticleItem(model, item);

    else if (item.modelType() == Constants::ParticleCoreShellType)
        populateParticleFromParticleItem(model, item);

    else if (item.modelType() == Constants::ParticleDistributionType)
        populateParticleFromParticleItem(model, item);

    else if (item.modelType() == Constants::MesoCrystalType)
        populateParticleFromParticleItem(model, item);
}

void RealSpaceBuilder::populateMultiLayer(RealSpaceModel* model, const SessionItem& item,
                                          const SceneGeometry& sceneGeometry, const QVector3D&)
{
    double total_height(0.0);
    int index(0);
    for (auto layer : item.getItems(MultiLayerItem::T_LAYERS)) {

        bool isTopLayer = index == 0 ? true : false;
        populateLayer(model, *layer, sceneGeometry,
                      QVector3D(0, 0, static_cast<float>(-total_height)), isTopLayer);

        if (index != 0)
            total_height += TransformTo3D::visualLayerThickness(*layer, sceneGeometry);

        ++index;
    }
}

void RealSpaceBuilder::populateLayer(RealSpaceModel* model, const SessionItem& layerItem,
                                     const SceneGeometry& sceneGeometry, const QVector3D& origin,
                                     const bool isTopLayer)
{
    auto layer = TransformTo3D::createLayer(layerItem, sceneGeometry, origin);
    if (layer && !isTopLayer)
        model->addBlend(layer.release());

    for (auto layout : layerItem.getItems(LayerItem::T_LAYOUTS))
        populateLayout(model, *layout, sceneGeometry, origin);
}

void RealSpaceBuilder::populateLayout(RealSpaceModel* model, const SessionItem& layoutItem,
                                      const SceneGeometry& sceneGeometry, const QVector3D& origin)
{
    Q_ASSERT(layoutItem.modelType() == Constants::ParticleLayoutType);

    // If there is no particle to populate
    if (!layoutItem.getItem(ParticleLayoutItem::T_PARTICLES))
        return;

    double layer_size = sceneGeometry.layer_size();
    double total_density = layoutItem.getItemValue(ParticleLayoutItem::P_TOTAL_DENSITY).toDouble();

    auto particle3DContainer_vector =
        RealSpaceBuilderUtils::particle3DContainerVector(layoutItem, origin);

    auto interference = GetInterferenceFunction(layoutItem);

    RealSpacePositionBuilder pos_builder;
    interference->accept(&pos_builder);
    std::vector<std::vector<double>> lattice_positions =
        pos_builder.generatePositions(layer_size, total_density);
    RealSpaceBuilderUtils::populateParticlesAtLatticePositions(
        lattice_positions, particle3DContainer_vector, model, sceneGeometry, this);
}

void RealSpaceBuilder::populateParticleFromParticleItem(RealSpaceModel* model,
                                                        const SessionItem& particleItem) const
{
    Particle3DContainer particle3DContainer;
    if (particleItem.modelType() == Constants::ParticleType) {
        auto pItem = dynamic_cast<const ParticleItem*>(&particleItem);
        auto particle = pItem->createParticle();
        particle3DContainer = RealSpaceBuilderUtils::singleParticle3DContainer(*particle);
    } else if (particleItem.modelType() == Constants::ParticleCoreShellType) {
        auto particleCoreShellItem = dynamic_cast<const ParticleCoreShellItem*>(&particleItem);
        // If there is no CORE or SHELL to populate inside ParticleCoreShellItem
        if (!particleCoreShellItem->getItem(ParticleCoreShellItem::T_CORE)
            || !particleCoreShellItem->getItem(ParticleCoreShellItem::T_SHELL))
            return;
        auto particleCoreShell = particleCoreShellItem->createParticleCoreShell();
        particle3DContainer =
            RealSpaceBuilderUtils::particleCoreShell3DContainer(*particleCoreShell);
    } else if (particleItem.modelType() == Constants::ParticleCompositionType) {
        auto particleCompositionItem = dynamic_cast<const ParticleCompositionItem*>(&particleItem);
        // If there is no particle to populate inside ParticleCompositionItem
        if (!particleCompositionItem->getItem(ParticleCompositionItem::T_PARTICLES))
            return;
        auto particleComposition = particleCompositionItem->createParticleComposition();
        particle3DContainer =
            RealSpaceBuilderUtils::particleComposition3DContainer(*particleComposition);
    } else if (particleItem.modelType() == Constants::ParticleDistributionType) {
        auto particleDistributionItem =
            dynamic_cast<const ParticleDistributionItem*>(&particleItem);
        // If there is no particle to populate inside ParticleDistributionItem
        if (!particleDistributionItem->getItem(ParticleDistributionItem::T_PARTICLES))
            return;
        // show nothing when ParticleDistributionItem is selected
    } else if (particleItem.modelType() == Constants::MesoCrystalType) {
        auto mesoCrystalItem = dynamic_cast<const MesoCrystalItem*>(&particleItem);
        // If there is no particle to populate inside MesoCrystalItem
        if (!mesoCrystalItem->getItem(MesoCrystalItem::T_BASIS_PARTICLE))
            return;
        particle3DContainer = RealSpaceBuilderUtils::mesoCrystal3DContainer(*mesoCrystalItem);
    }

    populateParticleFromParticle3DContainer(model, particle3DContainer);
}

void RealSpaceBuilder::populateParticleFromParticle3DContainer(
    RealSpaceModel* model, const Particle3DContainer& particle3DContainer,
    const QVector3D& lattice_position) const
{
    if (particle3DContainer.containerSize()) {
        for (size_t i = 0; i < particle3DContainer.containerSize(); ++i) {
            auto particle3D = particle3DContainer.createParticle(i);
            particle3D->addTranslation(lattice_position);
            if (particle3D) {
                if (!particle3DContainer.particle3DBlend(i))
                    model->add(particle3D.release());
                else
                    model->addBlend(particle3D.release()); // use addBlend() for transparent object
            }
        }
    }
}

namespace
{
std::unique_ptr<IInterferenceFunction> GetInterferenceFunction(const SessionItem& layoutItem)
{
    auto interferenceLattice = layoutItem.getItem(ParticleLayoutItem::T_INTERFERENCE);
    if (interferenceLattice) {
        auto interferenceItem = static_cast<const InterferenceFunctionItem*>(interferenceLattice);
        auto P_interference = interferenceItem->createInterferenceFunction();
        if (P_interference)
            return std::unique_ptr<IInterferenceFunction>(P_interference.release());
    }
    return std::make_unique<InterferenceFunctionNone>();
}
} // namespace
