// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Samples/inc/ISample.h
//! @brief     Defines interface class ISample.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2015
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   C. Durniak, M. Ganeva, G. Pospelov, W. Van Herck, J. Wuttke
//
// ************************************************************************** //

#ifndef ISAMPLE_H
#define ISAMPLE_H

#include "IParameterized.h"
#include "ICloneable.h"
#include "ISampleVisitor.h"

class ICompositeSample;
class DWBASimulation;

//! @class ISample
//! @ingroup samples_internal
//! @brief Interface for objects related to scattering.

class BA_CORE_API_ ISample : public ICloneable, public IParameterized
{
public:
    //! Returns pointer to "this", if it is a composite sample.
    virtual ICompositeSample *getCompositeSample() { return 0; }
    virtual const ICompositeSample *getCompositeSample() const { return 0; }

    //! Returns a clone of this ISample object
    virtual ISample *clone() const =0;

    //! Returns a clone with inverted magnetic fields
    virtual ISample *cloneInvertB() const;

    //! Calls the ISampleVisitor's visit method
    virtual void accept(ISampleVisitor *p_visitor) const = 0;

    //! Returns an ISimulation if DWBA is required.
    virtual DWBASimulation *createDWBASimulation() const { return 0; }

    virtual void printSampleTree();

    friend std::ostream& operator<<(std::ostream& ostr, const ISample& m)
    { m.print(ostr); return ostr; }

    virtual bool containsMagneticMaterial() const;

    //! Adds parameters from local pool to external pool and recursively calls its direct children.
    virtual std::string addParametersToExternalPool(std::string path, ParameterPool *external_pool,
                                                    int copy_number = -1) const;
};

#endif // ISAMPLE_H


