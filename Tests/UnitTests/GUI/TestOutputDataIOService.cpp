#include "ApplicationModels.h"
#include "DataItem.h"
#include "GUIHelpers.h"
#include "ImportDataInfo.h"
#include "IntensityDataIOFactory.h"
#include "JobItem.h"
#include "JobItemUtils.h"
#include "JobModel.h"
#include "JobModelFunctions.h"
#include "OutputData.h"
#include "OutputDataIOHistory.h"
#include "OutputDataIOService.h"
#include "ProjectUtils.h"
#include "RealDataItem.h"
#include "RealDataModel.h"
#include "google_test.h"
#include "test_utils.h"
#include <QTest>
#include <memory>

class TestOutputDataIOService : public ::testing::Test
{
public:
    TestOutputDataIOService();
    ~TestOutputDataIOService();

protected:
    OutputData<double> m_data;
};

TestOutputDataIOService::TestOutputDataIOService()
{
    FixedBinAxis axis0("x", 10, 0.0, 1.0);
    FixedBinAxis axis1("y", 10, -1.0, 1.0);
    m_data.addAxis(axis0);
    m_data.addAxis(axis1);
}

TestOutputDataIOService::~TestOutputDataIOService() = default;

//! Test methods for retrieving nonXML data.

TEST_F(TestOutputDataIOService, test_nonXMLData)
{
    ApplicationModels models;

    // initial state
    auto dataItems = models.nonXMLData();
    EXPECT_EQ(dataItems.size(), 0);

    // adding RealDataItem
    RealDataItem* realData =
        dynamic_cast<RealDataItem*>(models.realDataModel()->insertNewItem(Constants::RealDataType));
    EXPECT_EQ(models.realDataModel()->nonXMLData().size(), 0);
    realData->setOutputData(TestUtils::createData().release());
    EXPECT_EQ(models.realDataModel()->nonXMLData().size(), 1);

    // adding JobItem
    SessionItem* jobItem = models.jobModel()->insertNewItem(Constants::JobItemType);
    SessionItem* dataItem = models.jobModel()->insertNewItem(
        Constants::IntensityDataType, jobItem->index(), -1, JobItem::T_OUTPUT);
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 1);

    // adding RealDataItem to jobItem
    RealDataItem* realData2 = dynamic_cast<RealDataItem*>(models.jobModel()->insertNewItem(
        Constants::RealDataType, jobItem->index(), -1, JobItem::T_REALDATA));
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 1);
    realData2->setOutputData(TestUtils::createData(0.0, TestUtils::DIM::D1).release());
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 2);

    // checking data items of OutputDataIOService
    OutputDataIOService service(&models);
    EXPECT_EQ(service.nonXMLItems().size(), 3);

    // checking data items of ApplicationModels
    dataItems = models.nonXMLData();
    EXPECT_EQ(dataItems.size(), 3);
    EXPECT_EQ(dataItems.indexOf(realData->dataItem()), 0);
    EXPECT_EQ(dataItems.indexOf(dataItem), 1);
    EXPECT_EQ(dataItems.indexOf(realData2->dataItem()), 2);

    // Replacing the data inside RealDataItem with the data of the same dimensions
    realData->setOutputData(TestUtils::createData(2.0).release());
    EXPECT_EQ(models.realDataModel()->nonXMLData().size(), 1);

    // Replacing the data inside RealDataItem with the data of different dimensions
    auto data = TestUtils::createData(3.0, TestUtils::DIM::D1);
    EXPECT_THROW(dynamic_cast<RealDataItem*>(realData)->setOutputData(data.get()),
                 GUIHelpers::Error);
    EXPECT_EQ(models.realDataModel()->nonXMLData().size(), 1);
}

//! Tests OutputDataSaveInfo class intended for storing info about the last save.

TEST_F(TestOutputDataIOService, test_OutputDataSaveInfo)
{
    SessionModel model("TempModel");
    DataItem* item = dynamic_cast<DataItem*>(model.insertNewItem(Constants::IntensityDataType));

    item->setLastModified(QDateTime::currentDateTime());

    const int nap_time(10);
    QTest::qSleep(nap_time);

    OutputDataSaveInfo info = OutputDataSaveInfo::createSaved(item);
    EXPECT_TRUE(info.wasModifiedSinceLastSave() == false);

    QTest::qSleep(nap_time);
    item->setLastModified(QDateTime::currentDateTime());
    EXPECT_TRUE(info.wasModifiedSinceLastSave() == true);
}

//! Tests OutputDataDirHistory class intended for storing save history of several
//! IntensityDataItems in a directory.

TEST_F(TestOutputDataIOService, test_OutputDataDirHistory)
{
    SessionModel model("TempModel");
    DataItem* item1 = dynamic_cast<DataItem*>(model.insertNewItem(Constants::IntensityDataType));

    DataItem* item2 = dynamic_cast<DataItem*>(model.insertNewItem(Constants::IntensityDataType));
    item1->setOutputData(m_data.clone());
    item1->setLastModified(QDateTime::currentDateTime());
    item2->setLastModified(QDateTime::currentDateTime());

    // empty history
    OutputDataDirHistory history;
    EXPECT_TRUE(history.contains(item1) == false);
    // non-existing item is treated as modified
    EXPECT_TRUE(history.wasModifiedSinceLastSave(item1) == true);

    // Saving item in a history
    history.markAsSaved(item1);
    history.markAsSaved(item2);

    EXPECT_TRUE(history.contains(item1) == true);
    // Empty DataItems are not added to history:
    EXPECT_TRUE(history.contains(item2) == false);
    EXPECT_TRUE(history.wasModifiedSinceLastSave(item1) == false);

    // Attempt to save same item second time
    EXPECT_THROW(history.markAsSaved(item1), GUIHelpers::Error);

    // Modifying item
    QTest::qSleep(10);
    item1->setLastModified(QDateTime::currentDateTime());

    EXPECT_TRUE(history.wasModifiedSinceLastSave(item1) == true);
}

//! Tests OutputDataIOHistory class (save info for several independent directories).

TEST_F(TestOutputDataIOService, test_OutputDataIOHistory)
{
    SessionModel model("TempModel");
    DataItem* item1 = dynamic_cast<DataItem*>(model.insertNewItem(Constants::IntensityDataType));

    DataItem* item2 = dynamic_cast<DataItem*>(model.insertNewItem(Constants::IntensityDataType));
    item1->setOutputData(m_data.clone());
    item2->setOutputData(m_data.clone());

    item1->setLastModified(QDateTime::currentDateTime());
    item2->setLastModified(QDateTime::currentDateTime());

    OutputDataDirHistory dirHistory1;
    dirHistory1.markAsSaved(item1);
    dirHistory1.markAsSaved(item2);

    OutputDataDirHistory dirHistory2;
    dirHistory2.markAsSaved(item1);

    // Modifying item
    QTest::qSleep(10);
    item1->setLastModified(QDateTime::currentDateTime());

    // Creating two histories
    OutputDataIOHistory history;
    history.setHistory("dir1", dirHistory1);
    history.setHistory("dir2", dirHistory2);

    EXPECT_TRUE(history.wasModifiedSinceLastSave("dir1", item1) == true);
    EXPECT_TRUE(history.wasModifiedSinceLastSave("dir2", item1) == true);

    EXPECT_TRUE(history.wasModifiedSinceLastSave("dir1", item2) == false);
    EXPECT_TRUE(history.wasModifiedSinceLastSave("dir2", item2)
                == true); // since item2 doesn't exist

    // asking info for some non-existing directory
    EXPECT_THROW(history.wasModifiedSinceLastSave("dir3", item1), GUIHelpers::Error);
}

//! Testing saving abilities of OutputDataIOService class.

TEST_F(TestOutputDataIOService, test_OutputDataIOService)
{
    const QString projectDir("test_OutputDataIOService");
    TestUtils::create_dir(projectDir);

    const double value1(1.0), value2(2.0), value3(3.0);

    ApplicationModels models;
    RealDataItem* realData1 = TestUtils::createRealData("data1", *models.realDataModel(), value1);
    RealDataItem* realData2 = TestUtils::createRealData("data2", *models.realDataModel(), value2);

    // Saving first time
    OutputDataIOService service(&models);
    service.save(projectDir);
    QTest::qSleep(10);

    // Checking existance of data on disk
    QString fname1 = "./" + projectDir + "/" + "realdata_data1_0.int.gz";
    QString fname2 = "./" + projectDir + "/" + "realdata_data2_0.int.gz";
    EXPECT_TRUE(ProjectUtils::exists(fname1));
    EXPECT_TRUE(ProjectUtils::exists(fname2));

    // Reading data from disk, checking it is the same
    std::unique_ptr<OutputData<double>> dataOnDisk1(
        IntensityDataIOFactory::readOutputData(fname1.toStdString()));
    std::unique_ptr<OutputData<double>> dataOnDisk2(
        IntensityDataIOFactory::readOutputData(fname2.toStdString()));
    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk1, *realData1->dataItem()->getOutputData()));
    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk2, *realData2->dataItem()->getOutputData()));

    // Modifying data and saving the project.
    realData2->setOutputData(TestUtils::createData(value3).release());
    service.save(projectDir);
    QTest::qSleep(10);

    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk1, *realData1->dataItem()->getOutputData()));
    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk2, *realData2->dataItem()->getOutputData())
                == false);
    // checking that data on disk has changed
    dataOnDisk2.reset(IntensityDataIOFactory::readOutputData(fname2.toStdString()));
    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk2, *realData2->dataItem()->getOutputData()));

    // Renaming RealData and check that file on disk changed the name
    realData2->setItemName("data2new");
    service.save(projectDir);
    QTest::qSleep(10);

    QString fname2new = "./" + projectDir + "/" + "realdata_data2new_0.int.gz";
    EXPECT_TRUE(ProjectUtils::exists(fname2new));
    EXPECT_TRUE(TestUtils::isTheSame(fname2new, *realData2->dataItem()->getOutputData()));

    // Check that file with old name was removed.
    EXPECT_TRUE(ProjectUtils::exists(fname2) == false);
}

TEST_F(TestOutputDataIOService, test_RealDataItemWithNativeData)
{
    ApplicationModels models;

    // initial state
    auto dataItems = models.nonXMLData();
    EXPECT_EQ(dataItems.size(), 0);

    // adding RealDataItem
    RealDataItem* realData =
        dynamic_cast<RealDataItem*>(models.realDataModel()->insertNewItem(Constants::RealDataType));
    EXPECT_EQ(models.realDataModel()->nonXMLData().size(), 0);

    ImportDataInfo import_data(std::unique_ptr<OutputData<double>>(m_data.clone()),
                               Constants::UnitsNbins);
    realData->setImportData(std::move(import_data));

    EXPECT_EQ(models.realDataModel()->nonXMLData().size(), 2);
    realData->setItemValue(RealDataItem::P_NAME, QString("RealData"));

    // adding JobItem
    JobItem* jobItem =
        dynamic_cast<JobItem*>(models.jobModel()->insertNewItem(Constants::JobItemType));
    jobItem->setIdentifier(GUIHelpers::createUuid());
    models.jobModel()->insertNewItem(Constants::IntensityDataType, jobItem->index(), -1,
                                     JobItem::T_OUTPUT);
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 1);

    // copying RealDataItem to JobItem
    JobModelFunctions::copyRealDataItem(jobItem, realData);
    EXPECT_EQ(models.jobModel()->nonXMLData().size(), 3);

    // checking data items of OutputDataIOService
    OutputDataIOService service(&models);
    EXPECT_EQ(service.nonXMLItems().size(), 5);

    const QString projectDir("test_NativeData");
    TestUtils::create_dir(projectDir);

    // Saving
    service.save(projectDir);
    QTest::qSleep(10);

    // Checking existance of data on disk
    auto data1 = realData->dataItem();
    auto data2 = realData->nativeData();
    auto data3 = jobItem->realDataItem()->dataItem();
    auto data4 = jobItem->realDataItem()->nativeData();

    QString fname1 = "./" + projectDir + "/" + data1->fileName();
    QString fname2 = "./" + projectDir + "/" + data2->fileName();
    QString fname3 = "./" + projectDir + "/" + data3->fileName();
    QString fname4 = "./" + projectDir + "/" + data4->fileName();

    EXPECT_TRUE(ProjectUtils::exists(fname1));
    EXPECT_TRUE(ProjectUtils::exists(fname2));
    EXPECT_TRUE(ProjectUtils::exists(fname3));
    EXPECT_TRUE(ProjectUtils::exists(fname4));

    auto readData = [](const QString& filename) {
        return std::unique_ptr<OutputData<double>>(
            IntensityDataIOFactory::readOutputData(filename.toStdString()));
    };

    // Reading data from disk, checking it is the same
    auto dataOnDisk1 = readData(fname1);
    auto dataOnDisk2 = readData(fname2);
    auto dataOnDisk3 = readData(fname3);
    auto dataOnDisk4 = readData(fname4);
    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk1, *data1->getOutputData()));
    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk2, *data2->getOutputData()));
    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk3, *data3->getOutputData()));
    EXPECT_TRUE(TestUtils::isTheSame(*dataOnDisk4, *data4->getOutputData()));
}
