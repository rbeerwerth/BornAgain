// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/RealSpaceWidgets/RealSpaceBuilder.h
//! @brief     Defines class RealSpaceBuilder
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef REALSPACEBUILDER_H
#define REALSPACEBUILDER_H

#include "../../ba3d/ba3d/view/camera.h"
#include "WinDllMacros.h"
#include <QVector3D>
#include <QWidget>

class SessionItem;
class RealSpaceModel;
class Shape3D;
class SceneGeometry;
class Particle3DContainer;

class BA_CORE_API_ RealSpaceBuilder : public QWidget
{
    Q_OBJECT

public:
    RealSpaceBuilder(QWidget* parent = nullptr);

    ~RealSpaceBuilder();

    void populate(RealSpaceModel* model, const SessionItem& item,
                  const SceneGeometry& sceneGeometry,
                  const RealSpace::Camera::Position& cameraPosition =
                      RealSpace::Camera::Position(RealSpace::Vector3D(0, -200, 120), // eye
                                                  RealSpace::Vector3D(0, 0, 0),      // center
                                                  RealSpace::Vector3D::_z));         // up

    void populateMultiLayer(RealSpaceModel* model, const SessionItem& item,
                            const SceneGeometry& sceneGeometry,
                            const QVector3D& origin = QVector3D());

    void populateLayer(RealSpaceModel* model, const SessionItem& layerItem,
                       const SceneGeometry& sceneGeometry, const QVector3D& origin = QVector3D(),
                       const bool isTopLayer = false);

    void populateLayout(RealSpaceModel* model, const SessionItem& layoutItem,
                        const SceneGeometry& sceneGeometry, const QVector3D& origin = QVector3D());

    void populateParticleFromParticleItem(RealSpaceModel* model,
                                          const SessionItem& particleItem) const;

    void
    populateParticleFromParticle3DContainer(RealSpaceModel* model,
                                            const Particle3DContainer& particle3DContainer,
                                            const QVector3D& lattice_position = QVector3D()) const;
};

#endif // REALSPACEBUILDER_H
