#ifndef ROOTMINIMIZER_H
#define ROOTMINIMIZER_H
// ********************************************************************
// * The BornAgain project                                            *
// * Simulation of neutron and x-ray scattering at grazing incidence  *
// *                                                                  *
// * LICENSE AND DISCLAIMER                                           *
// * Lorem ipsum dolor sit amet, consectetur adipiscing elit.  Mauris *
// * eget quam orci. Quisque  porta  varius  dui,  quis  posuere nibh *
// * mollis quis. Mauris commodo rhoncus porttitor.                   *
// ********************************************************************
//! @file   ROOTMinimizer.h
//! @brief  Definition of ROOTMinimizer class
//! @author Scientific Computing Group at FRM II
//! @date   05.10.2012


#include "IMinimizer.h"
#include "OutputData.h"
#include "Exceptions.h"
#include "ROOTMinimizerFunction.h"
#include "FitSuiteParameters.h"
#include <string>
// from ROOT
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"


//- -------------------------------------------------------------------
//! @class ROOTMinimizer
//! @brief Wrapper for ROOT minimizers to interface with FitSuite
//- -------------------------------------------------------------------
class ROOTMinimizer : public IMinimizer
{
public:
    ROOTMinimizer(const std::string &minimizer_name, const std::string &algo_type=std::string());
    virtual ~ROOTMinimizer();

    virtual void setParameter(size_t index, const FitParameter *par);
    virtual void setParameters(const FitSuiteParameters &parameters);


    virtual void setFunction(function_chi2_t fun_chi2, size_t nparameters, function_gradient_t fun_gradient, size_t ndatasize);

    virtual void minimize();

    //! return created minimizer
    ROOT::Math::Minimizer *getROOTMinimizer() { return m_root_minimizer; }

    //! get number of variables to fit
    virtual size_t getNumberOfVariables() const { return m_root_minimizer->NDim(); }

    //! return minimum function value
    virtual double getMinValue() const { return m_root_minimizer->MinValue(); }

    //! return value of variable corresponding the minimum of the function
    virtual double getValueOfVariableAtMinimum(size_t i) const {
        if(i >= getNumberOfVariables() ) throw OutOfBoundsException("ROOTMinimizer::getValueOfVariableAtMinimum() -> Wrong number of the variable");
        return m_root_minimizer->X()[i];
    }

    //! return value of variable corresponding the minimum of the function
    virtual double getErrorOfVariable(size_t i) const {
        if(i >= getNumberOfVariables() ) throw OutOfBoundsException("ROOTMinimizer::getErrorOfVariable() -> Wrong number of the variable");
        return (m_root_minimizer->Errors() == 0? 0 : m_root_minimizer->Errors()[i]);
    }

    //! printing results
    virtual void printResults() const;

    //! clear resources (parameters) for consecutives minimizations
    virtual void clear() { m_root_minimizer->Clear(); }

    //! printing minimizer description
    virtual void printOptions() const;

    //! checking validity of the combination minimizer_name and algo_type
    bool isValidNames(const std::string &minimizer_name, const std::string &algo_type);

    //! check if type of algorithm is Levenberg-Marquardt or similar
    bool isGradientBasedAgorithm();

private:
    std::string m_minimizer_name;
    std::string m_algo_type;

    ROOT::Math::Minimizer *m_root_minimizer;
    ROOTMinimizerFunction * m_minfunc;
    ROOTMinimizerElementFunction * m_minfunc_element;

//    IMinimizerFunction *m_minimizer_function;
//    ROOT::Math::Functor *m_fcn; //! function to minimize
//    ROOT::Math::GradFunctor *m_fcn_grad; //! gradient of function to minimize
//    function_t m_fcn;
//    element_function_t m_element_fcn;
//    int m_ndims;
//    int m_nelements;
};

#endif // ROOTMINIMIZER_H
