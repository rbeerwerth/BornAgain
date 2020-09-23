// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/HardParticle/IFormFactorPrism.h
//! @brief     Defines class FormFactorPolygonalPrism.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef BORNAGAIN_CORE_HARDPARTICLE_IFORMFACTORPRISM_H
#define BORNAGAIN_CORE_HARDPARTICLE_IFORMFACTORPRISM_H

#include "Core/HardParticle/PolyhedralComponents.h"
#include "Core/HardParticle/PolyhedralTopology.h"
#include "Core/Scattering/IFormFactorBorn.h"
#include <memory>

//! A prism with a polygonal base, for form factor computation.

class BA_CORE_API_ FormFactorPolygonalPrism : public IFormFactorBorn
{
public:
    FormFactorPolygonalPrism() = default;
    FormFactorPolygonalPrism(const NodeMeta& meta, const std::vector<double>& PValues);

    double bottomZ(const IRotation& rotation) const override final;
    double topZ(const IRotation& rotation) const override final;

    virtual complex_t evaluate_for_q(cvector_t q) const override;
    virtual double volume() const override;
    double getHeight() const { return height(); }
    virtual double radialExtension() const override { return std::sqrt(m_base->area()); }

protected:
    virtual double height() const = 0;
    std::unique_ptr<PolyhedralFace> m_base;
    void setPrism(bool symmetry_Ci, const std::vector<kvector_t>& vertices);
    std::vector<kvector_t> m_vertices; //! for topZ, bottomZ computation only
};

//! A polygonal surface, for testing form factor computations.

class BA_CORE_API_ FormFactorPolygonalSurface : public IFormFactorBorn
{
public:
    FormFactorPolygonalSurface() = default;
    FormFactorPolygonalSurface(const NodeMeta& meta, const std::vector<double>& PValues);

    complex_t evaluate_for_q(cvector_t q) const override final;
    double volume() const override { return 0; }
    double radialExtension() const override final { return std::sqrt(m_base->area()); }

protected:
    std::unique_ptr<PolyhedralFace> m_base;
};

#endif // BORNAGAIN_CORE_HARDPARTICLE_IFORMFACTORPRISM_H
