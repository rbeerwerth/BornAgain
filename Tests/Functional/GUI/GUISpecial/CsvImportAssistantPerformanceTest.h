// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Tests/Functional/GUI/GUISpecial/CsvImportAssistantPerformanceTest.h
//! @brief     Defines GUI performance functional test.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef CSVIMPORTASSISTANTPERFORMANCETEST_H
#define CSVIMPORTASSISTANTPERFORMANCETEST_H

#include "IFunctionalTest.h"
#include "SpecularDataItem.h"
#include <QString>
#include <memory>

//! Functional test to measure performance of CsvImportAssistant by loading files of different sizes
class CsvImportAssistantPerformanceTest : public IFunctionalTest
{
public:
    CsvImportAssistantPerformanceTest();
    ~CsvImportAssistantPerformanceTest();

protected:
    const QString testFilename() { return QString::fromStdString(m_testFilename); }
    void writeTestFile();
    void writeTestFile(size_t nRows, size_t nCols);
    OutputData<double>* readTestFile();
    bool runTest();

    const std::string m_testFilename = "tmp_TestCsvImportAssistant.txt";
    const std::vector<std::vector<double>> m_testVector = {
        {0.0, 1.0, 2.0, 3.0},     {4.0, 5.0, 6.0, 7.0},     {8.0, 9.0, 10.0, 11.0},
        {12.0, 13.0, 14.0, 15.0}, {16.0, 17.0, 18.0, 19.0}, {20.0, 21.0, 22.0, 23.0}};

private:
};

#endif // CSVIMPORTASSISTANTPERFORMANCETEST_H
