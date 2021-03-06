#include "ApplicationModels.h"
#include "DataItem.h"
#include "GUIHelpers.h"
#include "GroupItem.h"
#include "InstrumentItems.h"
#include "InstrumentModel.h"
#include "IntensityDataIOFactory.h"
#include "ItemFileNameUtils.h"
#include "JobItem.h"
#include "JobModel.h"
#include "JobModelFunctions.h"
#include "OutputData.h"
#include "OutputDataIOService.h"
#include "PointwiseAxis.h"
#include "PointwiseAxisItem.h"
#include "ProjectUtils.h"
#include "SpecularBeamInclinationItem.h"
#include "google_test.h"
#include "test_utils.h"
#include <QTest>

class TestSavingSpecularData : public ::testing::Test
{
public:
    TestSavingSpecularData();
    ~TestSavingSpecularData();

protected:
    SpecularInstrumentItem* createSpecularInstrument(ApplicationModels& models);
    PointwiseAxisItem* createPointwiseAxisItem(SessionModel& model);
    GroupItem* getAxisGroup(SpecularInstrumentItem* instrument);
    bool isSame(const QString& filename, const IAxis* axis);

    std::unique_ptr<PointwiseAxis> m_axis;
};

TestSavingSpecularData::TestSavingSpecularData()
    : m_axis(new PointwiseAxis("x", std::vector<double>{0.1, 0.2, 1.0}))
{
}

TestSavingSpecularData::~TestSavingSpecularData() = default;

SpecularInstrumentItem* TestSavingSpecularData::createSpecularInstrument(ApplicationModels& models)
{
    return dynamic_cast<SpecularInstrumentItem*>(
        models.instrumentModel()->insertNewItem(Constants::SpecularInstrumentType));
}

PointwiseAxisItem* TestSavingSpecularData::createPointwiseAxisItem(SessionModel& model)
{
    auto instrument_item = dynamic_cast<SpecularInstrumentItem*>(
        model.insertNewItem(Constants::SpecularInstrumentType));
    return dynamic_cast<PointwiseAxisItem*>(
        getAxisGroup(instrument_item)->getChildOfType(Constants::PointwiseAxisType));
}

GroupItem* TestSavingSpecularData::getAxisGroup(SpecularInstrumentItem* instrument)
{
    auto axis_item = instrument->getItem(SpecularInstrumentItem::P_BEAM)
                         ->getItem(BeamItem::P_INCLINATION_ANGLE)
                         ->getItem(SpecularBeamInclinationItem::P_ALPHA_AXIS);
    return dynamic_cast<GroupItem*>(axis_item);
}

bool TestSavingSpecularData::isSame(const QString& filename, const IAxis* axis)
{
    std::unique_ptr<OutputData<double>> dataOnDisk(
        IntensityDataIOFactory::readOutputData(filename.toStdString()));
    OutputData<double> refData;
    refData.addAxis(*axis);
    return TestUtils::isTheSame(*dataOnDisk, refData);
}

TEST_F(TestSavingSpecularData, test_SpecularInsturment)
{
    ApplicationModels models;

    // initial state
    auto dataItems = models.nonXMLData();
    EXPECT_EQ(dataItems.size(), 0);

    // adding instrument
    auto instrument = createSpecularInstrument(models);

    // instrument contains hidden pointwise axis item
    EXPECT_EQ(models.instrumentModel()->nonXMLData().size(), 1);

    // explicitly switching to pointwise axis item
    auto axis_group = getAxisGroup(instrument);
    axis_group->setCurrentType(Constants::PointwiseAxisType);
    EXPECT_EQ(models.instrumentModel()->nonXMLData().size(), 1);

    // hiding pointwise axis item back
    axis_group->setCurrentType(Constants::BasicAxisType);
    EXPECT_EQ(models.instrumentModel()->nonXMLData().size(), 1);

    // checking data items of OutputDataIOService
    OutputDataIOService service(&models);
    EXPECT_EQ(service.nonXMLItems().size(), 1);

    // checking data items of ApplicationModels
    dataItems = models.nonXMLData();
    EXPECT_EQ(dataItems.size(), 1);
}

TEST_F(TestSavingSpecularData, test_InstrumentInJobItem)
{
    ApplicationModels models;

    // adding JobItem
    SessionItem* jobItem = models.jobModel()->insertNewItem(Constants::JobItemType);
    SessionItem* dataItem = models.jobModel()->insertNewItem(
        Constants::IntensityDataType, jobItem->index(), -1, JobItem::T_OUTPUT);
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 1);

    // adding instrument
    auto instrument = dynamic_cast<SpecularInstrumentItem*>(models.jobModel()->insertNewItem(
        Constants::SpecularInstrumentType, jobItem->index(), -1, JobItem::T_INSTRUMENT));
    // instrument contains hidden pointwise axis item
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 2);

    // explicitly switching to pointwise axis item
    auto axis_group = getAxisGroup(instrument);
    axis_group->setCurrentType(Constants::PointwiseAxisType);
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 2);

    OutputDataIOService service(&models);
    EXPECT_EQ(service.nonXMLItems().size(), 2);

    auto dataItems = models.nonXMLData();
    EXPECT_EQ(dataItems.size(), 2);
    EXPECT_EQ(dataItems.indexOf(dataItem), 0);

    // hiding pointwise axis, should be saved anyway
    axis_group->setCurrentType(Constants::BasicAxisType);
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 2);
    EXPECT_EQ(service.nonXMLItems().size(), 2);
    dataItems = models.nonXMLData();
    EXPECT_EQ(dataItems.size(), 2);
    EXPECT_EQ(dataItems.indexOf(dataItem), 0);
}

TEST_F(TestSavingSpecularData, test_setLastModified)
{
    SessionModel model("TempModel");
    auto item = createPointwiseAxisItem(model);

    const int nap_time(10);
    QTest::qSleep(nap_time);

    OutputDataSaveInfo info = OutputDataSaveInfo::createSaved(item);
    EXPECT_FALSE(info.wasModifiedSinceLastSave());

    QTest::qSleep(nap_time);
    item->init(*m_axis.get(), Constants::UnitsDegrees);
    EXPECT_TRUE(info.wasModifiedSinceLastSave());

    info = OutputDataSaveInfo::createSaved(item);
    QTest::qSleep(nap_time);
    item->setItemValue(PointwiseAxisItem::P_FILE_NAME, QString("new_value"));
    EXPECT_TRUE(info.wasModifiedSinceLastSave());
}

TEST_F(TestSavingSpecularData, test_DirHistory)
{
    SessionModel model("TempModel");
    auto item1 = createPointwiseAxisItem(model);
    item1->init(*m_axis, Constants::UnitsDegrees);

    auto item2 = createPointwiseAxisItem(model);

    // empty history
    OutputDataDirHistory history;
    EXPECT_FALSE(history.contains(item1));
    // non-existing item is treated as modified
    EXPECT_TRUE(history.wasModifiedSinceLastSave(item1));

    // Saving item in a history
    history.markAsSaved(item1);
    history.markAsSaved(item2);

    EXPECT_TRUE(history.contains(item1));
    // Empty DataItems are not added to history:
    EXPECT_FALSE(history.contains(item2));
    EXPECT_FALSE(history.wasModifiedSinceLastSave(item1));

    // Attempt to save same item second time
    EXPECT_THROW(history.markAsSaved(item1), GUIHelpers::Error);

    // Modifying item
    QTest::qSleep(10);
    item1->init(*m_axis, Constants::UnitsDegrees);

    EXPECT_TRUE(history.wasModifiedSinceLastSave(item1));
}

//! Testing saving abilities of OutputDataIOService class.
TEST_F(TestSavingSpecularData, test_OutputDataIOService)
{
    const QString projectDir("test_SpecularDataSave");
    TestUtils::create_dir(projectDir);

    // setting up items and data

    ApplicationModels models;
    auto instrument1 = createSpecularInstrument(models);
    auto instrument2 = createSpecularInstrument(models);

    auto axis_group1 = getAxisGroup(instrument1);
    auto pointwise_axis_item1 =
        dynamic_cast<PointwiseAxisItem*>(axis_group1->getChildOfType(Constants::PointwiseAxisType));
    pointwise_axis_item1->init(*m_axis, Constants::UnitsDegrees);

    auto axis_group2 = getAxisGroup(instrument2);
    auto pointwise_axis_item2 =
        dynamic_cast<PointwiseAxisItem*>(axis_group2->getChildOfType(Constants::PointwiseAxisType));
    PointwiseAxis tmp("y", std::vector<double>{1.0, 2.0, 3.0});
    pointwise_axis_item2->init(tmp, Constants::UnitsRadians);

    // Saving first time
    OutputDataIOService service(&models);
    service.save(projectDir);
    QTest::qSleep(10);

    // Checking existance of data on disk
    QString fname1 = "./" + projectDir + "/" + pointwise_axis_item1->fileName();
    QString fname2 = "./" + projectDir + "/" + pointwise_axis_item2->fileName();
    EXPECT_TRUE(ProjectUtils::exists(fname1));
    EXPECT_TRUE(ProjectUtils::exists(fname2));

    // Reading data from disk, checking it is the same
    EXPECT_TRUE(isSame(fname1, pointwise_axis_item1->getAxis()));
    EXPECT_TRUE(isSame(fname2, pointwise_axis_item2->getAxis()));
    ;

    // Modifying data and saving the project.
    PointwiseAxis tmp2("z", std::vector<double>{2.0, 3.0, 4.0});
    pointwise_axis_item2->init(tmp2, Constants::UnitsRadians);
    service.save(projectDir);
    QTest::qSleep(10);

    EXPECT_TRUE(isSame(fname1, pointwise_axis_item1->getAxis()));
    EXPECT_TRUE(isSame(fname2, pointwise_axis_item2->getAxis()));

    // Renaming RealData and check that file on disk changed the name
    pointwise_axis_item2->setItemValue(PointwiseAxisItem::P_FILE_NAME, "data2new.int.gz");
    service.save(projectDir);
    QTest::qSleep(10);

    QString fname2new = "./" + projectDir + "/" + pointwise_axis_item2->fileName();
    EXPECT_TRUE(ProjectUtils::exists(fname2new));
    EXPECT_TRUE(isSame(fname2new, pointwise_axis_item2->getAxis()));

    // Check that file with old name was removed.
    EXPECT_FALSE(ProjectUtils::exists(fname2));
}

TEST_F(TestSavingSpecularData, test_CopyInstrumentToJobItem)
{
    const QString projectDir("test_SpecularDataSave2");
    TestUtils::create_dir(projectDir);

    ApplicationModels models;

    // adding instrument and initializing pointwise axis
    auto instrument = createSpecularInstrument(models);
    auto axis_group = getAxisGroup(instrument);
    auto pointwise_axis_item =
        dynamic_cast<PointwiseAxisItem*>(axis_group->getChildOfType(Constants::PointwiseAxisType));
    pointwise_axis_item->init(*m_axis, Constants::UnitsQyQz);

    // adding JobItem and copying instrument
    auto jobItem = dynamic_cast<JobItem*>(models.jobModel()->insertNewItem(Constants::JobItemType));
    JobModelFunctions::setupJobItemInstrument(jobItem, instrument);
    auto job_instrument =
        dynamic_cast<SpecularInstrumentItem*>(jobItem->getItem(JobItem::T_INSTRUMENT));
    auto job_axis_item = dynamic_cast<PointwiseAxisItem*>(
        getAxisGroup(job_instrument)->getChildOfType(Constants::PointwiseAxisType));

    // checking filenames
    EXPECT_EQ(pointwise_axis_item->fileName(),
              ItemFileNameUtils::instrumentDataFileName(*instrument));
    EXPECT_EQ(job_axis_item->fileName(),
              ItemFileNameUtils::instrumentDataFileName(*job_instrument));

    // Saving
    OutputDataIOService service(&models);
    service.save(projectDir);
    QTest::qSleep(10);

    // Checking existance of data on disk
    QString fname1 = "./" + projectDir + "/" + pointwise_axis_item->fileName();
    QString fname2 = "./" + projectDir + "/" + job_axis_item->fileName();
    EXPECT_FALSE(fname1 == fname2);
    EXPECT_TRUE(ProjectUtils::exists(fname1));
    EXPECT_TRUE(ProjectUtils::exists(fname2));

    // Reading data from disk, checking it is the same
    EXPECT_TRUE(isSame(fname1, pointwise_axis_item->getAxis()));
    EXPECT_TRUE(isSame(fname2, job_axis_item->getAxis()));
}
