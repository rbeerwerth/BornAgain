// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/HardParticle/FormFactorBar.h
//! @brief     Defines classes FormFactorBar*.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef BORNAGAIN_CORE_HARDPARTICLE_FORMFACTORBAR_H
#define BORNAGAIN_CORE_HARDPARTICLE_FORMFACTORBAR_H

#include "Core/HardParticle/ProfileBar.h"

//! The form factor of an elongated bar, with Gaussian profile in elongation direction.
//! @ingroup legacyGrating
class BA_CORE_API_ FormFactorBarGauss : public ProfileBar
{
public:
    FormFactorBarGauss(double length, double width, double height);
    FormFactorBarGauss* clone() const override final;
    void accept(INodeVisitor* visitor) const override final;

private:
    complex_t factor_x(complex_t qx) const override final;
};

//! The form factor of an elongated, with Lorentz form factor in elongation direction.
//! @ingroup legacyGrating
class BA_CORE_API_ FormFactorBarLorentz : public ProfileBar
{
public:
    FormFactorBarLorentz(double length, double width, double height);
    FormFactorBarLorentz* clone() const override final;
    void accept(INodeVisitor* visitor) const override final;

private:
    complex_t factor_x(complex_t qx) const override final;
};

#endif // BORNAGAIN_CORE_HARDPARTICLE_FORMFACTORBAR_H
