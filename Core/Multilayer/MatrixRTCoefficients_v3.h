// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file     Core/Multilayer/MatrixRTCoefficients_v3.h
//! @brief    Defines class MatrixRTCoefficients_v3.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
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

    Eigen::Matrix4cd getM() const override {return MM + MS;}
//    Eigen::Matrix4cd getML() const override {return ML;}
    Eigen::Matrix4cd getMM() const override {return MM;}
    Eigen::Matrix4cd getMS() const override {return MS;}

    Eigen::Matrix2cd getReflectionMatrix() const override;

private:
    double m_kz_sign; //! wave propagation direction (-1 for direct one, 1 for time reverse)
    Eigen::Vector2cd m_lambda; //!< eigenvalues for wave propagation
    kvector_t m_b; //!< normalized magnetic field impact (with correction for external mag. field)
    double m_magnetic_SLD;

//    Eigen::Vector4cd m_w_plus; //!< boundary values for up-polarization
//    Eigen::Vector4cd m_w_min;  //!< boundary values for down-polarization

//    Eigen::Matrix4cd T1; //!< matrix selecting the transmitted part of
//                         !< the first eigenmode
//    Eigen::Matrix4cd R1; //!< matrix selecting the reflected part of
//                         !< the first eigenmode
//    Eigen::Matrix4cd T2; //!< matrix selecting the transmitted part of
//                         !< the second eigenmode
//    Eigen::Matrix4cd R2; //!< matrix selecting the reflected part of
                         //!< the second eigenmode
                         //!
                         //!
    // new structures
    Eigen::Vector4cd m_t_r_plus;
    Eigen::Vector4cd m_t_r_minus;

    Eigen::Matrix4cd MiL;
    Eigen::Matrix4cd MiS;

    Eigen::Matrix4cd MM;
    Eigen::Matrix4cd MS;

    // helper functions to compute DWBA compatible amplitudes
    Eigen::Matrix2cd T1Matrix() const;
    Eigen::Matrix2cd T2Matrix() const;

    Eigen::Matrix2cd R1Matrix() const;
    Eigen::Matrix2cd R2Matrix() const;
};

#endif // MATRIXRTCOEFFICIENTS_V3_H
