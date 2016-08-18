// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/StandardSamples/TwoDimLatticeBuilder.h
//! @brief     Defines class IsGISAXS06Builder.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2015
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   C. Durniak, M. Ganeva, G. Pospelov, W. Van Herck, J. Wuttke
//
// ************************************************************************** //

#ifndef TWODIMLATTICEBUILDER_H
#define TWODIMLATTICEBUILDER_H

#include "ISampleBuilder.h"

//! Builds sample: 2D lattice with different disorder (IsGISAXS example #6).
//! @ingroup standard_samples

class BA_CORE_API_ SquareLatticeBuilder : public ISampleBuilder
{
public:
    SquareLatticeBuilder(){}
    MultiLayer* buildSample() const;
};


//! Builds sample: 2D lattice with different disorder (IsGISAXS example #6).
//! @ingroup standard_samples

class BA_CORE_API_ CenteredSquareLatticeBuilder : public ISampleBuilder
{
public:
    CenteredSquareLatticeBuilder(){}
    MultiLayer* buildSample() const;
};


//! Builds sample: 2D lattice with different disorder (IsGISAXS example #6).
//! @ingroup standard_samples

class BA_CORE_API_ RotatedSquareLatticeBuilder : public ISampleBuilder
{
public:
    RotatedSquareLatticeBuilder(){}
    MultiLayer* buildSample() const;
};

#endif // TWODIMLATTICEBUILDER_H
