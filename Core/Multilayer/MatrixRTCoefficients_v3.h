// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file     Core/Multilayer/MatrixRTCoefficients_v3.h
//! @brief    Defines class MatrixRTCoefficients_v3.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2020
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef MATRIXRTCOEFFICIENTS_V3_H
#define MATRIXRTCOEFFICIENTS_V3_H

#include "ILayerRTCoefficients.h"
#include "Vectors3D.h"
#include <vector>

//! Specular reflection and transmission coefficients in a layer in case
//! of magnetic interactions between the scattered particle and the layer.
//! @ingroup algorithms_internal

class BA_CORE_API_ MatrixRTCoefficients_v3 : public ILayerRTCoefficients
{
public:
    friend class SpecularMagneticNewStrategy;

    MatrixRTCoefficients_v3(double kz_sign, Eigen::Vector2cd eigenvalues, kvector_t b, double magnetic_SLD);
    MatrixRTCoefficients_v3(const MatrixRTCoefficients_v3& other);
    ~MatrixRTCoefficients_v3() override;

    MatrixRTCoefficients_v3* clone() const override;

    //! The following functions return the transmitted and reflected amplitudes
    //! for different incoming beam polarizations and eigenmodes
    Eigen::Vector2cd T1plus() const override;
    Eigen::Vector2cd R1plus() const override;
    Eigen::Vector2cd T2plus() const override;
    Eigen::Vector2cd R2plus() const override;
    Eigen::Vector2cd T1min() const override;
    Eigen::Vector2cd R1min() const override;
    Eigen::Vector2cd T2min() const override;
    Eigen::Vector2cd R2min() const override;
    //! Returns z-part of the two wavevector eigenmodes
    Eigen::Vector2cd getKz() const override;

    Eigen::Matrix2cd pMatrixHelper(double sign) const;
    Eigen::Matrix2cd computeP() const;
    Eigen::Matrix2cd computeInverseP() const;

    std::pair<Eigen::Matrix2cd, Eigen::Matrix2cd> computeDeltaMatrix(double thickness, double prefactor);

    Eigen::Matrix2cd getReflectionMatrix() const override;

private:
    double m_kz_sign; //! wave propagation direction (-1 for direct one, 1 for time reverse)
    Eigen::Vector2cd m_lambda; //!< eigenvalues for wave propagation
    kvector_t m_b; //!< unit magnetic field vector
    double m_magnetic_SLD;

    Eigen::Vector4cd m_t_r_plus;  // amplitudes for incoming up-polarization
    Eigen::Vector4cd m_t_r_minus; // amplitudes for incoming down-polarization

    Eigen::Matrix4cd m_MiL; // Large part of the backwards transfer matrix
                            // between current and next layer
    Eigen::Matrix4cd m_MiS; // small part of the backwards transfer matrix
                            // between current and next layer

    Eigen::Matrix4cd m_ML; // Large part of the total backwards transfer matrix
    Eigen::Matrix4cd m_MS; // Small part of the total backwards transfer matrix

    // helper functions to compute DWBA compatible amplitudes used in the T1plus() etc. functions
    Eigen::Matrix2cd TransformationMatrix(complex_t eigenvalue, Eigen::Vector2d selection) const;
    Eigen::Matrix2cd T1Matrix() const;
    Eigen::Matrix2cd T2Matrix() const;
};

#endif // MATRIXRTCOEFFICIENTS_V3_H
