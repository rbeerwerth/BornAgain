// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Tests/Functional/Core/SelfConsistenceTest/SelfConsistenceTest.h
//! @brief     Defines class SelfConsistenceTest.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef BORNAGAIN_TESTS_FUNCTIONAL_CORE_SELFCONSISTENCETEST_SELFCONSISTENCETEST_H
#define BORNAGAIN_TESTS_FUNCTIONAL_CORE_SELFCONSISTENCETEST_SELFCONSISTENCETEST_H

#include <memory>
#include <string>
#include <vector>

class Simulation;

//! A functional test of BornAgain/Core.
//! Performs given simulations and compares their results with each other.

class SelfConsistenceTest
{
public:
    SelfConsistenceTest(const std::string& name,
                        std::vector<std::unique_ptr<Simulation>> simulations, double threshold);
    ~SelfConsistenceTest() = default;

    bool runTest();

private:
    std::string m_name;
    std::vector<std::unique_ptr<Simulation>> m_simulations;
    double m_threshold;
};

#endif // BORNAGAIN_TESTS_FUNCTIONAL_CORE_SELFCONSISTENCETEST_SELFCONSISTENCETEST_H
