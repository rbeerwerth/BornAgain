// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Tests/Functional/Std/StandardTestService.cpp
//! @brief     Defines class StandardTestService.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "Tests/Functional/Std/StandardTestService.h"
#include "Core/Multilayer/IMultiLayerBuilder.h"
#include "Core/Multilayer/MultiLayer.h"
#include "Core/StandardSamples/SampleBuilderFactory.h"
#include "Core/StandardSamples/SimulationFactory.h"
#include "Tests/Functional/Std/IStandardTest.h"
#include "Tests/Functional/Std/StandardTestCatalog.h"
#include <cassert>
#include <iostream>

namespace
{

//! Returns test full name, composed from the test name and the name of sample builder.

std::string fullTestName(const std::string& test_name, const std::string& builder_name)
{
    std::string result = test_name;

    if (builder_name != "SampleBuilder")
        result += "_" + builder_name;

    return result;
}

} // namespace

//! Runs test corresponding to given command line parameters, returns true if success.

bool StandardTestServiceBase::execute(int argc, char** argv)
{
    assert(argc > 1);
    StandardTestInfo info = StandardTestCatalog().testInfo(argv[1]);
    assert(!info.m_test_name.empty());
    assert(info.size() == 1);

    std::unique_ptr<IMultiLayerBuilder> builder(
        SampleBuilderFactory().createItem(info.m_sample_builder_names.front()));

    size_t n_subtests = builder->size();
    int number_of_failed_tests = 0;

    for (size_t sample_index = 0; sample_index < builder->size(); ++sample_index) {
        std::cout << "Run std test " << info.m_test_name << ", subtest " << sample_index << "/"
                  << builder->size() << ": " << builder->getName() << "\n";

        std::unique_ptr<Simulation> simulation(
            SimulationFactory().createItem(info.m_simulation_names.front()));

        std::unique_ptr<MultiLayer> sample(builder->createSample(sample_index));
        simulation->setSample(*sample);

        std::string test_name = fullTestName(info.m_test_name, builder->getName());

        std::unique_ptr<IStandardTest> test(
            createStdTest(test_name, *simulation, info.m_threshold));

        if (!test->runTest()) {
            std::cout << "subtest " << sample_index << " failed\n";
            ++number_of_failed_tests;
        }
    }

    if (number_of_failed_tests)
        std::cout << "summary: " << number_of_failed_tests << " of " << n_subtests
                  << " subtests failed\n";

    return number_of_failed_tests == 0;
}
