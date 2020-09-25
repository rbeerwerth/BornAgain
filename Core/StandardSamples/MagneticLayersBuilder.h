// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/StandardSamples/MagneticLayersBuilder.h
//! @brief    Defines class to build samples with magnetic layers
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef BORNAGAIN_CORE_STANDARDSAMPLES_MAGNETICLAYERSBUILDER_H
#define BORNAGAIN_CORE_STANDARDSAMPLES_MAGNETICLAYERSBUILDER_H

#include "Core/SampleBuilderEngine/ISampleBuilder.h"
#include "Core/Multilayer/RoughnessModels.h"

class ISample;

//! Builds sample: spheres in substrate layer with a zero magnetic field.
//! @ingroup standard_samples

class MagneticSubstrateZeroFieldBuilder : public ISampleBuilder
{
public:
    MagneticSubstrateZeroFieldBuilder();
    MultiLayer* buildSample() const;

private:
    double m_sphere_radius;
};

//! Builds sample: ambient and one magnetized layer on a non-magnetized substrate.
//! @ingroup standard_samples

class SimpleMagneticLayerBuilder : public ISampleBuilder
{
public:
    SimpleMagneticLayerBuilder();
    MultiLayer* buildSample() const;
};

//! Builds sample: magnetic spheres in a magnetized layer on a non-magnetized substrate.
//! @ingroup standard_samples

class MagneticLayerBuilder : public ISampleBuilder
{
public:
    MagneticLayerBuilder();
    MultiLayer* buildSample() const;

private:
    double m_sphere_radius;
};

//! Builds sample: magnetic layer on a magnetic substrate with the fields rotated by 90°
//! @ingroup standard_samples

class SimpleMagneticRotationBuilder : public ISampleBuilder
{
public:
    SimpleMagneticRotationBuilder();
    MultiLayer* buildSample() const override;

    MultiLayer* createSample(size_t index) override;
    size_t size() override;

protected:
    MultiLayer* builder(double sigmaRoughness = 0.,
                        RoughnessModel roughnessModel = RoughnessModel::TANH) const;
};

//! Builds sample: rotated magnetic spheres in substrate layer with a unit magnetic field.
//! @ingroup standard_samples

class MagneticRotationBuilder : public ISampleBuilder
{
public:
    MagneticRotationBuilder();
    MultiLayer* buildSample() const;

private:
    double m_sphere_radius;
};

#endif // BORNAGAIN_CORE_STANDARDSAMPLES_MAGNETICLAYERSBUILDER_H
