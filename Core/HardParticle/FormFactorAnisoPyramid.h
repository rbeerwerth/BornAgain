// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/HardParticle/FormFactorAnisoPyramid.h
//! @brief     Defines class FormFactorAnisoPyramid
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef FORMFACTORANISOPYRAMID_H
#define FORMFACTORANISOPYRAMID_H

#include "FormFactorPolyhedron.h"

//! A frustum (truncated pyramid) with rectangular base.
//! @ingroup hardParticle

class BA_CORE_API_ FormFactorAnisoPyramid : public FormFactorPolyhedron
{
public:
    FormFactorAnisoPyramid(double length, double width, double height, double alpha);

    FormFactorAnisoPyramid* clone() const override final
    {
        return new FormFactorAnisoPyramid(m_length, m_width, m_height, m_alpha);
    }
    void accept(INodeVisitor* visitor) const override final { visitor->visit(this); }

    double getLength() const { return m_length; }
    double getWidth() const { return m_width; }
    double getHeight() const { return m_height; }
    double getAlpha() const { return m_alpha; }

protected:
    IFormFactor* sliceFormFactor(ZLimits limits, const IRotation& rot,
                                 kvector_t translation) const override final;

    void onChange() override final;

private:
    static const PolyhedralTopology topology;

    double m_length;
    double m_width;
    double m_height;
    double m_alpha;
};

#endif // FORMFACTORANISOPYRAMID_H
