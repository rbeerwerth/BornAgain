// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Particle/IClusteredParticles.h
//! @brief     Defines class IClusteredParticles.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2015
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   C. Durniak, M. Ganeva, G. Pospelov, W. Van Herck, J. Wuttke
//
// ************************************************************************** //

#ifndef ICLUSTEREDPARTICLES_H
#define ICLUSTEREDPARTICLES_H

#include "ISample.h"
#include "Vectors3D.h"

class IFormFactor;
class IRotation;

//! An ordered assembly of particles. Currently, the only child class is Crystal.
//! @ingroup samples_internal

class BA_CORE_API_ IClusteredParticles : public ISample
{
public:
    IClusteredParticles() {}

    virtual IClusteredParticles* clone() const =0;
    virtual IClusteredParticles* cloneInvertB() const =0;

    virtual void accept(INodeVisitor* visitor) const =0;

    virtual void setAmbientMaterial(const IMaterial& material) =0;
    virtual const IMaterial* getAmbientMaterial() const =0;

    //! Creates a total form factor for the mesocrystal with a specific shape and content
    //! The bulk content of the mesocrystal is encapsulated by the IClusteredParticles object itself
    virtual IFormFactor* createTotalFormFactor(
        const IFormFactor&, const IRotation*, const kvector_t& /*translation*/) const =0;

    //! Composes transformation with existing one
    virtual void applyRotation(const IRotation&) =delete;
};

#endif // ICLUSTEREDPARTICLES_H
