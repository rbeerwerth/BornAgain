// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      FormFactors/FormFactorTetrahedron.h
//! @brief     Declares class FormFactorTetrahedron
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2015-16
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   C. Durniak, M. Ganeva, G. Pospelov, W. Van Herck, J. Wuttke
//
// ************************************************************************** //

#ifndef FORMFACTORTETRAHEDRON_H
#define FORMFACTORTETRAHEDRON_H

#include "FormFactorPolyhedron.h"

//! @class FormFactorTetrahedron
//! @ingroup formfactors
//! @brief The formfactor of tetrahedron.

class BA_CORE_API_ FormFactorTetrahedron : public FormFactorPolyhedron
{
public:
    FormFactorTetrahedron(double length, double height, double alpha);

    virtual FormFactorTetrahedron *clone() const;
    virtual void accept(ISampleVisitor *visitor) const;

    double getLength() const { return m_length; }
    double getHeight() const { return m_height; }
    double getAlpha() const { return m_alpha; }

private:
    virtual void onChange() final;
    double m_length;
    double m_height;
    double m_alpha;
};

#endif // FORMFACTORTETRAHEDRON_H
