// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Parametrization/RealParameter.h
//! @brief     Defines class RealParameter.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef REALPARAMETER_H
#define REALPARAMETER_H

#include "Attributes.h"
#include "IParameter.h"
#include "RealLimits.h"
#include "Unit.h"
#include <string>

class ParameterPool;

//! Wraps a parameter of type double.
//! In addition to name and on-change callback held by the parent class IParameter,
//! this class holds Limits, Attributes (currently only fixed or not), and a Unit.
//! @ingroup tools_internal

class BA_CORE_API_ RealParameter : public IParameter<double>
{
public:
    RealParameter(const std::string& name, double* par,
                  const std::string& parent_name = std::string(),
                  const std::function<void()>& onChange = std::function<void()>(),
                  const RealLimits& limits = RealLimits::limitless(),
                  const Attributes& attr = Attributes::free());

    RealParameter* clone(const std::string& new_name = "") const;

    //! Sets value of wrapped parameter and emit signal
    void setValue(double value);

    //! Returns value of wrapped parameter
    double value() const;

    RealParameter& setLimits(const RealLimits& limits);
    RealLimits limits() const;

    RealParameter& setLimited(double lower, double upper);
    RealParameter& setPositive();
    RealParameter& setNonnegative();

    RealParameter& setUnit(const std::string& name);
    std::string unit() const;

protected:
    RealLimits m_limits;
    Attributes m_attr;
    Unit m_unit;
};

#endif // REALPARAMETER_H
