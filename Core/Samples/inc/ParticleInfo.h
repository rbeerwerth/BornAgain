// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Samples/inc/ParticleInfo.h
//! @brief     Defines class ParticleInfo.
//!
//! @homepage  http://apps.jcns.fz-juelich.de/BornAgain
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2013
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   C. Durniak, G. Pospelov, W. Van Herck, J. Wuttke 
//
// ************************************************************************** //

#ifndef PARTICLEINFO_H
#define PARTICLEINFO_H

#include "ICompositeSample.h"
#include "Particle.h"
#include "Transform3D.h"

//! Holds additional information about particle (used in ParticleDecoration)

class ParticleInfo : public ICompositeSample
{
  public:
    ParticleInfo(Particle *p_particle,
                 Geometry::Transform3D *transform=0,
                 double depth=0, double abundance=0);
    ParticleInfo(const Particle& p_particle,
                 const Geometry::Transform3D& transform,
                 double depth=0, double abundance=0);
    virtual ~ParticleInfo();

    virtual ParticleInfo *clone() const;

    //! Return particle.
    const Particle *getParticle() const { return mp_particle; }

    //! Return transformation.
    const Geometry::Transform3D *getTransform3D() const { return mp_transform; }

    //! Set transformation.
    void setTransform(const Geometry::Transform3D &transform) {
        delete mp_transform;
        mp_transform = new Geometry::Transform3D(transform);
    }

    //! Return depth.
    double getDepth() const { return m_depth;}

    //! Set depth.
    void setDepth(double depth) { m_depth = depth; }

    //! Return abundance.
    double getAbundance() const { return m_abundance; }

    //! Set abundance.
    void setAbundance(double abundance) { m_abundance = abundance; }

protected:
    //! register some class members for later access via parameter pool
    virtual void init_parameters();

    Particle *mp_particle;
    Geometry::Transform3D *mp_transform;
    double m_depth;
    double m_abundance;
};

#endif /* PARTICLEINFO_H */
