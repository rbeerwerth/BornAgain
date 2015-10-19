// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      coregui/Models/TransformToDomain.cpp
//! @brief     Implements class TransformToDomain
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2015
//! @authors   Scientific Computing Group at MLZ Garching
//! @authors   C. Durniak, M. Ganeva, G. Pospelov, W. Van Herck, J. Wuttke
//
// ************************************************************************** //

#include "TransformToDomain.h"
#include "MaterialUtils.h"
#include "GUIHelpers.h"
#include "FormFactorItems.h"
#include "InterferenceFunctionItems.h"
#include "ParticleItem.h"
#include "LayerItem.h"
#include "BeamItem.h"
#include "ComboProperty.h"
#include "DetectorItems.h"
#include "AxesItems.h"
#include "MultiLayerItem.h"
#include "LatticeTypeItems.h"
#include "FTDistributionItems.h"
#include "ParticleCoreShellItem.h"
#include "ParticleCoreShell.h"
#include "LayerRoughnessItems.h"
#include "VectorItem.h"
#include "MaterialUtils.h"
#include "MaterialProperty.h"
#include "AngleProperty.h"
#include "FixedBinAxis.h"
#include "ConstKBinAxis.h"
#include "ParticleLayoutItem.h"
#include "DistributionItem.h"
#include "BeamWavelengthItem.h"
#include "BeamAngleItems.h"
#include "ResolutionFunctionItems.h"
#include <QDebug>

#include <boost/scoped_ptr.hpp>


IMaterial *TransformToDomain::createDomainMaterial(const ParameterizedItem &item)
{
    MaterialProperty material_property;
    if(item.modelType() == Constants::ParticleType) {
        material_property = item.getRegisteredProperty(ParticleItem::P_MATERIAL).value<MaterialProperty>();
    }
    else if(item.modelType() == Constants::LayerType) {
        material_property = item.getRegisteredProperty(LayerItem::P_MATERIAL).value<MaterialProperty>();
    }
    if(!material_property.isDefined())
        throw GUIHelpers::Error("TransformToDomain::createDomainMaterial() -> Error. Unknown item to create material");

    return MaterialUtils::createDomainMaterial(material_property);
}

MultiLayer *TransformToDomain::createMultiLayer(const ParameterizedItem &item)
{
    MultiLayer *result = new MultiLayer();
    double cross_corr_length =
            item.getRegisteredProperty(
                MultiLayerItem::P_CROSS_CORR_LENGTH).toDouble();
    if(cross_corr_length>0) result->setCrossCorrLength(cross_corr_length);
    result->setName(item.itemName().toUtf8().constData());
    return result;
}

Layer *TransformToDomain::createLayer(const ParameterizedItem &item)
{
    Layer *result = new Layer();
    double thickness =
            item.getRegisteredProperty(LayerItem::P_THICKNESS).toDouble();
    result->setThickness(thickness);

    boost::scoped_ptr<IMaterial> P_material(createDomainMaterial(item));
    result->setMaterial(*P_material.get());
    result->setName(item.itemName().toUtf8().constData());

    return result;
}

ParticleLayout *TransformToDomain::createParticleLayout(
        const ParameterizedItem &item)
{
    (void)item;
    ParticleLayout *result = new ParticleLayout();
    QVariant var = item.getRegisteredProperty(ParticleLayoutItem::P_APPROX);
    ComboProperty prop = var.value<ComboProperty>();
    QString approximation = prop.getValue();
    if (approximation == QString("Decoupling Approximation")) {
        result->setApproximation(ILayout::DA);
    }
    else if (approximation == QString("Size Space Coupling Approximation")) {
        result->setApproximation(ILayout::SSCA);
    }
    double total_density =
            item.getRegisteredProperty(ParticleLayoutItem::P_TOTAL_DENSITY).value<double>();
    result->setTotalParticleSurfaceDensity(total_density);
    return result;
}

Particle *TransformToDomain::createParticle(const ParameterizedItem &item, double &abundance)
{
    boost::scoped_ptr<IMaterial> P_material(createDomainMaterial(item));
    Particle *result = new Particle(*P_material);

    PropertyAttribute attribute = item.getPropertyAttribute(ParticleItem::P_ABUNDANCE);
//    if(attribute.getAppearance() == PropertyAttribute::DISABLED) {
//        throw GUIHelpers::Error("TransformToDomain::createParticle() -> Logic Error? "
//            "You are trying to get the value of DISABLED abundancy for model "+ item.modelType());
//    }

    abundance = item.getRegisteredProperty(ParticleItem::P_ABUNDANCE).toDouble();
    result->setName(item.itemName().toStdString());

    ParameterizedItem *ffItem = item.getSubItems()[ParticleItem::P_FORM_FACTOR];
    Q_ASSERT(ffItem);

    boost::scoped_ptr<IFormFactor> P_ff(createFormFactor(*ffItem));
    result->setFormFactor(*P_ff);

    return result;
}

IFormFactor *TransformToDomain::createFormFactor(const ParameterizedItem &item)
{
    const FormFactorItem *ffItem = dynamic_cast<const FormFactorItem *>(&item);
    Q_ASSERT(ffItem);
    return ffItem->createFormFactor();
}

IDistribution1D *TransformToDomain::createDistribution(
        const ParameterizedItem &item)
{
    const DistributionItem *distr_item =
            dynamic_cast<const DistributionItem *>(&item);
    Q_ASSERT(distr_item);
    return distr_item->createDistribution();
}

IInterferenceFunction *TransformToDomain::createInterferenceFunction(
        const ParameterizedItem &item)
{
    if(item.modelType() == Constants::InterferenceFunctionRadialParaCrystalType) {
        double peak_distance = item.getRegisteredProperty(
                    InterferenceFunctionRadialParaCrystalItem::P_PEAK_DISTANCE)
                .toDouble();
        double damping_length = item.getRegisteredProperty(
                    InterferenceFunctionRadialParaCrystalItem::P_DAMPING_LENGTH)
                .toDouble();
        double domain_size = item.getRegisteredProperty(
                    InterferenceFunctionRadialParaCrystalItem::P_DOMAIN_SIZE)
                .toDouble();
        double kappa = item.getRegisteredProperty(
                    InterferenceFunctionRadialParaCrystalItem::P_KAPPA)
                .toDouble();

        InterferenceFunctionRadialParaCrystal *result =
                new InterferenceFunctionRadialParaCrystal(peak_distance,
                                                      damping_length);
        result->setDomainSize(domain_size);
        result->setKappa(kappa);
        ParameterizedItem *pdfItem = item.getSubItems()[
                InterferenceFunctionRadialParaCrystalItem::P_PDF];

        Q_ASSERT(pdfItem);
        boost::scoped_ptr<IFTDistribution1D> P_pdf(
                    dynamic_cast<FTDistribution1DItem *>(pdfItem)
                    ->createFTDistribution());
        Q_ASSERT(P_pdf.get());

        result->setProbabilityDistribution(*P_pdf);
        return result;
    }
    else if(item.modelType() == Constants::InterferenceFunction2DParaCrystalType) {

        ParameterizedItem *latticeItem = item.getSubItems()
                [InterferenceFunction2DParaCrystalItem::P_LATTICE_TYPE];
        Q_ASSERT(latticeItem);

        double length_1(0), length_2(0), alpha_lattice(0.0);
        if(latticeItem->modelType() == Constants::BasicLatticeType) {
            length_1 = latticeItem->getRegisteredProperty(
                        BasicLatticeTypeItem::P_LATTICE_LENGTH1).toDouble();
            length_2 = latticeItem->getRegisteredProperty(
                        BasicLatticeTypeItem::P_LATTICE_LENGTH2).toDouble();
            alpha_lattice = Units::deg2rad(latticeItem->getRegisteredProperty(
                        BasicLatticeTypeItem::P_LATTICE_ANGLE).toDouble());
        }
        else if(latticeItem->modelType() == Constants::SquareLatticeType) {
            length_1 = latticeItem->getRegisteredProperty(
                        SquareLatticeTypeItem::P_LATTICE_LENGTH).toDouble();
            length_2 = length_1;
            alpha_lattice = Units::PI/2.0;
        }
        else if(latticeItem->modelType() == Constants::HexagonalLatticeType) {
            length_1 = latticeItem->getRegisteredProperty(
                        HexagonalLatticeTypeItem::P_LATTICE_LENGTH).toDouble();
            length_2 = length_1;
            alpha_lattice = 2*Units::PI/3.0;
        }
        else {
            throw GUIHelpers::Error("TransformToDomain::createInterferenceFunction() -> Error");
        }

        InterferenceFunction2DParaCrystal *result = new InterferenceFunction2DParaCrystal(
                    length_1,
                    length_2,
                    alpha_lattice,
                    Units::deg2rad(item.getRegisteredProperty(InterferenceFunction2DParaCrystalItem::P_ROTATION_ANGLE).toDouble()),
                    item.getRegisteredProperty(InterferenceFunction2DParaCrystalItem::P_DAMPING_LENGTH).toDouble());
        result->setDomainSizes(
                    item.getRegisteredProperty(InterferenceFunction2DParaCrystalItem::P_DOMAIN_SIZE1).toDouble(),
                    item.getRegisteredProperty(InterferenceFunction2DParaCrystalItem::P_DOMAIN_SIZE2).toDouble()
                    );

        result->setIntegrationOverXi(item.getRegisteredProperty(InterferenceFunction2DParaCrystalItem::P_XI_INTEGRATION).toBool());

        ParameterizedItem *pdf1Item = item.getSubItems()[InterferenceFunction2DParaCrystalItem::P_PDF1];
        Q_ASSERT(pdf1Item);
        boost::scoped_ptr<IFTDistribution2D> P_pdf1(dynamic_cast<FTDistribution2DItem *>(pdf1Item)->createFTDistribution());
        Q_ASSERT(P_pdf1.get());

        ParameterizedItem *pdf2Item = item.getSubItems()[InterferenceFunction2DParaCrystalItem::P_PDF2];
        Q_ASSERT(pdf2Item);
        boost::scoped_ptr<IFTDistribution2D> P_pdf2(dynamic_cast<FTDistribution2DItem *>(pdf2Item)->createFTDistribution());
        Q_ASSERT(P_pdf2.get());

        result->setProbabilityDistributions(*P_pdf1, *P_pdf2);
        return result;
    }
    else if(item.modelType() == Constants::InterferenceFunction2DLatticeType) {

        ParameterizedItem *latticeItem = item.getSubItems()
                [InterferenceFunction2DLatticeItem::P_LATTICE_TYPE];
        Q_ASSERT(latticeItem);

        double length_1, length_2, angle;
        if(latticeItem->modelType() == Constants::BasicLatticeType) {
            length_1 = latticeItem->getRegisteredProperty(
                        BasicLatticeTypeItem::P_LATTICE_LENGTH1).toDouble();
            length_2 = latticeItem->getRegisteredProperty(
                        BasicLatticeTypeItem::P_LATTICE_LENGTH2).toDouble();
            angle = Units::deg2rad(latticeItem->getRegisteredProperty(
                        BasicLatticeTypeItem::P_LATTICE_ANGLE).toDouble());
        }
        else if(latticeItem->modelType() == Constants::SquareLatticeType) {
            length_1 = latticeItem->getRegisteredProperty(
                        SquareLatticeTypeItem::P_LATTICE_LENGTH).toDouble();
            length_2 = length_1;
            angle = Units::PI/2.0;
        }
        else if(latticeItem->modelType() == Constants::HexagonalLatticeType) {
            length_1 = latticeItem->getRegisteredProperty(
                        HexagonalLatticeTypeItem::P_LATTICE_LENGTH).toDouble();
            length_2 = length_1;
            angle = 2*Units::PI/3.0;
        }
        else {
            throw GUIHelpers::Error("TransformToDomain::createInterferenceFunction() -> Error");
        }
        double xi = Units::deg2rad(item.getRegisteredProperty(
            InterferenceFunction2DLatticeItem::P_ROTATION_ANGLE).toDouble());

        InterferenceFunction2DLattice *result =
                new InterferenceFunction2DLattice(length_1, length_2, angle, xi);

        ParameterizedItem *pdfItem = item.getSubItems()
                [InterferenceFunction2DLatticeItem::P_PDF];
        Q_ASSERT(pdfItem);
        boost::scoped_ptr<IFTDistribution2D> P_pdf(
             dynamic_cast<FTDistribution2DItem *>(pdfItem)
                    ->createFTDistribution());
        Q_ASSERT(P_pdf.get());

        result->setProbabilityDistribution(*P_pdf);
        return result;
    }

    return 0;
}

Instrument *TransformToDomain::createInstrument(const ParameterizedItem &item)
{
//    qDebug() << "TransformToDomain::createInstrument";
    Instrument *result = new Instrument();
    result->setName(item.itemName().toUtf8().constData());
    return result;
}

Beam *TransformToDomain::createBeam(const ParameterizedItem &item)
{
//    qDebug() << "TransformToDomain::createBeam";
    Beam *result = new Beam();
    result->setName(item.itemName().toUtf8().constData());

    const BeamItem *beamItem = dynamic_cast<const BeamItem *>(&item);

    result->setIntensity(beamItem->getIntensity());
    double lambda = beamItem->getWavelength();
    double inclination_angle = Units::deg2rad(beamItem->getInclinationAngle());
    double azimuthal_angle = Units::deg2rad(beamItem->getAzimuthalAngle());
    result->setCentralK( lambda, inclination_angle, azimuthal_angle);

    return result;
}

void TransformToDomain::initInstrumentFromDetectorItem(const ParameterizedItem &item, Instrument *instrument)
{
    ParameterizedItem *subDetector = item.getSubItems()[DetectorItem::P_DETECTOR];
    Q_ASSERT(subDetector);

    if (subDetector->modelType() == Constants::PhiAlphaDetectorType) {

        BasicAxisItem *x_axis = dynamic_cast<BasicAxisItem *>(subDetector->getSubItems()[PhiAlphaDetectorItem::P_PHI_AXIS]);
        Q_ASSERT(x_axis);
        int n_x = x_axis->getRegisteredProperty(BasicAxisItem::P_NBINS).toInt();
        double x_min = Units::deg2rad(x_axis->getRegisteredProperty(BasicAxisItem::P_MIN).toDouble());
        double x_max = Units::deg2rad(x_axis->getRegisteredProperty(BasicAxisItem::P_MAX).toDouble());

        BasicAxisItem *y_axis = dynamic_cast<BasicAxisItem *>(subDetector->getSubItems()[PhiAlphaDetectorItem::P_ALPHA_AXIS]);
        Q_ASSERT(y_axis);
        int n_y = y_axis->getRegisteredProperty(BasicAxisItem::P_NBINS).toInt();
        double y_min = Units::deg2rad(y_axis->getRegisteredProperty(BasicAxisItem::P_MIN).toDouble());
        double y_max = Units::deg2rad(y_axis->getRegisteredProperty(BasicAxisItem::P_MAX).toDouble());

        instrument->setDetectorParameters(n_x, x_min, x_max, n_y, y_min, y_max);

        // setting up resolution function
        ResolutionFunctionItem *resfuncItem = dynamic_cast<ResolutionFunctionItem *>
                (subDetector->getSubItems()[PhiAlphaDetectorItem::P_RESOLUTION_FUNCTION]);
        Q_ASSERT(resfuncItem);

        IResolutionFunction2D *resfunc = resfuncItem->createResolutionFunction();
        if(resfunc)
            instrument->setDetectorResolutionFunction(*resfunc);
        delete resfunc;

    }
    else {
        throw GUIHelpers::Error("TransformToDomain::initInstrumentWithDetectorItem() -> Error. Unknown model type "+subDetector->modelType());
    }

}


ParticleCoreShell *TransformToDomain::createParticleCoreShell(const ParameterizedItem &item,
    const Particle &core, const Particle &shell, double &abundance)
{
    abundance = item.getRegisteredProperty(ParticleItem::P_ABUNDANCE).toDouble();

    ParticleCoreShell *result = new ParticleCoreShell(shell, core);
    result->setName(item.itemName().toStdString());
    return result;
}


ParticleComposition *TransformToDomain::createParticleComposition(const ParameterizedItem &item,
    double &abundance)
{
    abundance = item.getRegisteredProperty(ParticleItem::P_ABUNDANCE).toDouble();
    ParticleComposition *result = new ParticleComposition();
    return result;
}


LayerRoughness *TransformToDomain::createLayerRoughness(const ParameterizedItem &roughnessItem)
{
    if(roughnessItem.modelType() == Constants::LayerZeroRoughnessType) {
        return 0;
    }
    else if(roughnessItem.modelType() == Constants::LayerBasicRoughnessType) {
        LayerRoughness *result = new LayerRoughness(
                    roughnessItem.getRegisteredProperty(LayerBasicRoughnessItem::P_SIGMA).toDouble(),
                    roughnessItem.getRegisteredProperty(LayerBasicRoughnessItem::P_HURST).toDouble(),
                    roughnessItem.getRegisteredProperty(LayerBasicRoughnessItem::P_LATERAL_CORR_LENGTH).toDouble()
                    );
        return result;
    }
    else {
        throw GUIHelpers::Error("TransformToDomain::createLayerROughness() -> Error.");
    }

}

//! adds DistributionParameters to the Simulation
void TransformToDomain::addDistributionParametersToSimulation(const ParameterizedItem &beam_item, GISASSimulation *simulation)
{
    if(beam_item.modelType() == Constants::BeamType) {

        if(BeamWavelengthItem *beamWavelength = dynamic_cast<BeamWavelengthItem *>(beam_item.getSubItems()[BeamItem::P_WAVELENGTH])) {
            ParameterDistribution *distr = beamWavelength->getParameterDistributionForName("*/Beam/wavelength");
            if(distr) simulation->addParameterDistribution(*distr);
            delete distr;
        }

        if(BeamInclinationAngleItem *inclinationAngle = dynamic_cast<BeamInclinationAngleItem *>(beam_item.getSubItems()[BeamItem::P_INCLINATION_ANGLE])) {
            ParameterDistribution *distr = inclinationAngle->getParameterDistributionForName("*/Beam/alpha");
            if(distr) simulation->addParameterDistribution(*distr);
            delete distr;
        }

        if(BeamAzimuthalAngleItem *azimuthalAngle = dynamic_cast<BeamAzimuthalAngleItem *>(beam_item.getSubItems()[BeamItem::P_AZIMUTHAL_ANGLE])) {
            ParameterDistribution *distr = azimuthalAngle->getParameterDistributionForName("*/Beam/phi");
            if(distr) simulation->addParameterDistribution(*distr);
            delete distr;
        }
    }

}
