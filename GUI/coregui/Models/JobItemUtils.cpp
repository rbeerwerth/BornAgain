// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Models/JobItemUtils.cpp
//! @brief     Implements class JobItemUtils
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#include "JobItemUtils.h"
#include "ComboProperty.h"
#include "DomainObjectBuilder.h"
#include "GUIHelpers.h"
#include "InstrumentItems.h"
#include "DataItem.h"
#include "JobItem.h"
#include "RealDataItem.h"
#include "Simulation.h"
#include "UnitConverterUtils.h"
#include <QFileInfo>
#include <QDebug>

namespace
{
QMap<QString, AxesUnits> init_units_to_description_map()
{
    QMap<QString, AxesUnits> result;
    result[Constants::UnitsNbins] = AxesUnits::NBINS;
    result[Constants::UnitsRadians] = AxesUnits::RADIANS;
    result[Constants::UnitsDegrees] = AxesUnits::DEGREES;
    result[Constants::UnitsMm] = AxesUnits::MM;
    result[Constants::UnitsQyQz] = AxesUnits::QSPACE;
    return result;
}

QMap<AxesUnits, QString> init_description_to_units_map()
{
    QMap<AxesUnits, QString> result;
    result[AxesUnits::NBINS] = Constants::UnitsNbins;
    result[AxesUnits::RADIANS] = Constants::UnitsRadians;
    result[AxesUnits::DEGREES] = Constants::UnitsDegrees;
    result[AxesUnits::MM] = Constants::UnitsMm;
    result[AxesUnits::QSPACE] = Constants::UnitsQyQz;
    return result;
}

//! Updates axes' titles
void updateAxesTitle(DataItem* intensityItem, const IUnitConverter& converter, AxesUnits units);
}

//! Updates axes of OutputData in IntensityData item to correspond with ::P_AXES_UNITS selection.
//! InstrumentItem is used to get domain's detector map for given units.

void JobItemUtils::updateDataAxes(DataItem* intensityItem,
                                  const InstrumentItem* instrumentItem)
{
    Q_ASSERT(intensityItem);

    if (!instrumentItem) {
        // special case when reading old project files: project failed on load instrument
        // but we want to try to recover the rest.
        qInfo() << "JobItemUtils::updateDataAxes() -> Error. Absent instrument.";
        return;
    }

    if (!intensityItem->getOutputData())
        return;

    AxesUnits requested_units
        = axesUnitsFromName(intensityItem->selectedAxesUnits());

    const auto converter = DomainObjectBuilder::createUnitConverter(instrumentItem);
    auto newData = UnitConverterUtils::createOutputData(*converter.get(), requested_units);
    newData->setRawDataVector(intensityItem->getOutputData()->getRawDataVector());

    intensityItem->setOutputData(newData.release());
    intensityItem->setAxesRangeToData();
    updateAxesTitle(intensityItem, *converter, requested_units);
}

//! Correspondance of domain detector axes types to their gui counterpart.

QString JobItemUtils::nameFromAxesUnits(AxesUnits units)
{
    static QMap<AxesUnits, QString> units_to_name = init_description_to_units_map();
    return units_to_name[units];
}

//! Correspondance of GUI axes units names to their domain counterpart.

AxesUnits JobItemUtils::axesUnitsFromName(const QString& name)
{
    static QMap<QString, AxesUnits> name_to_units = init_units_to_description_map();
    return name_to_units[name];
}

//! Sets axes units suitable for given instrument.

void JobItemUtils::setIntensityItemAxesUnits(DataItem* intensityItem,
                                              const InstrumentItem* instrumentItem)
{
    const auto converter = DomainObjectBuilder::createUnitConverter(instrumentItem);
    setIntensityItemAxesUnits(intensityItem, *converter);
}

void JobItemUtils::setIntensityItemAxesUnits(DataItem *intensityItem,
                                             const IUnitConverter& converter)
{
    ComboProperty combo = availableUnits(converter);
    intensityItem->setItemValue(DataItem::P_AXES_UNITS, combo.variant());
}

void JobItemUtils::createDefaultDetectorMap(DataItem* intensityItem,
                                            const InstrumentItem* instrumentItem)
{
    const auto converter = DomainObjectBuilder::createUnitConverter(instrumentItem);
    auto output_data = UnitConverterUtils::createOutputData(*converter, converter->defaultUnits());
    intensityItem->setOutputData(output_data.release());
    setIntensityItemAxesUnits(intensityItem, *converter);
    updateAxesTitle(intensityItem, *converter, converter->defaultUnits());
}

void JobItemUtils::setResults(DataItem* intensityItem, const Simulation* simulation)
{
    const auto sim_result = simulation->result();
    if (intensityItem->getOutputData() == nullptr) {
        const auto& converter = sim_result.converter();
        JobItemUtils::setIntensityItemAxesUnits(intensityItem, converter);
        updateAxesTitle(intensityItem, converter, converter.defaultUnits());
    }
    auto selected_units = JobItemUtils::axesUnitsFromName(intensityItem->selectedAxesUnits());
    std::unique_ptr<OutputData<double>> data(sim_result.data(selected_units));
    intensityItem->setOutputData(data.release());
}

ComboProperty JobItemUtils::availableUnits(const IUnitConverter& converter)
{
    ComboProperty result;
    for (auto units : converter.availableUnits())
        result << nameFromAxesUnits(units);

    result.setValue(nameFromAxesUnits(converter.defaultUnits()));
    return result;
}

namespace
{
void updateAxesTitle(DataItem* intensityItem, const IUnitConverter& converter,
                     AxesUnits units)
{
    intensityItem->setXaxisTitle(QString::fromStdString(converter.axisName(0, units)));
    if (converter.dimension() > 1)
        intensityItem->setYaxisTitle(QString::fromStdString(converter.axisName(1, units)));
}
}
