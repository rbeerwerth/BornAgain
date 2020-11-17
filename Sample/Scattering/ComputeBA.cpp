//  ************************************************************************************************
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Sample/Scattering/ComputeBA.cpp
//! @brief     Implements class ComputeBA.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
//  ************************************************************************************************

#include "Sample/Scattering/ComputeBA.h"
#include "Sample/Material/WavevectorInfo.h"

ComputeBA::ComputeBA(const IFormFactor& ff) : IComputeFF(ff)
{
    setName("ComputeBA");
}

ComputeBA::~ComputeBA() = default;

ComputeBA* ComputeBA::clone() const
{
    return new ComputeBA(*m_ff);
}

complex_t ComputeBA::evaluate(const WavevectorInfo& wavevectors) const
{
    return m_ff->evaluate(wavevectors);
}
