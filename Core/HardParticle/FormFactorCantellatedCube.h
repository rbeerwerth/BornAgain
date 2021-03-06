// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/HardParticle/FormFactorCantellatedCube.h
//! @brief     Defines class FormFactorCantellatedCube.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef FORMFACTOREDGETRUNCATEDCUBE_H
#define FORMFACTOREDGETRUNCATEDCUBE_H

#include "FormFactorPolyhedron.h"

//! A cube, with truncation of all edges and corners, as in Croset (2017) Fig 7
//! @ingroup hardParticle

class BA_CORE_API_ FormFactorCantellatedCube : public FormFactorPolyhedron
{
public:
    FormFactorCantellatedCube(double length, double removed_length);

    FormFactorCantellatedCube* clone() const override final
    {
        return new FormFactorCantellatedCube(m_length, m_removed_length);
    }
    void accept(INodeVisitor* visitor) const override final { visitor->visit(this); }

    double getLength() const { return m_length; }
    double getRemovedLength() const { return m_removed_length; }

protected:
    void onChange() override final;

private:
    static const PolyhedralTopology topology;
    double m_length;
    double m_removed_length;
};

#endif // FORMFACTOREDGETRUNCATEDCUBE_H
