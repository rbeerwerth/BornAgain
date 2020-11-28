//  ************************************************************************************************
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      Core/Export/SampleLabelHandler.cpp
//! @brief     Implement class SampleLabelHandler.
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
//  ************************************************************************************************

#include "Core/Export/SampleLabelHandler.h"
#include "Sample/Multilayer/MultiLayer.h"
#include <set>

std::string SampleLabelHandler::labelMaterial(const Material* mat) {
    return m_MaterialLabel[mat];
}

std::string SampleLabelHandler::labelMultiLayer(const MultiLayer* ml) {
    return m_MultiLayerLabel[ml];
}

std::string SampleLabelHandler::labelRotation(const IRotation* rot) {
    return m_RotationsLabel[rot];
}

void SampleLabelHandler::insertMaterial(const Material* mat) {
    std::string label = "material_" + std::to_string(m_MaterialLabel.size() + 1);
    m_MaterialLabel.insert(mat, label);
}

void SampleLabelHandler::insertMultiLayer(const MultiLayer* sample) {
    std::string label = "multiLayer_" + std::to_string(m_MultiLayerLabel.size() + 1);
    m_MultiLayerLabel.insert(sample, label);
}

void SampleLabelHandler::insertRotation(const IRotation* sample) {
    std::string label = "rotation_" + std::to_string(m_RotationsLabel.size() + 1);
    m_RotationsLabel.insert(sample, label);
}



void SampleLabelHandler::insertKeyedObject(const std::string& key, const INode* s) {
    m_objects[key].emplace_back(s);
}

std::string SampleLabelHandler::obj2key(const INode* s) const {
    for (auto it: m_objects) {
        const std::vector<const INode*>& v = it.second;
        const auto vpos = std::find(v.begin(), v.end(), s);
        if (vpos == std::end(v))
            continue;
        const std::string& key = it.first;
        if (v.size() == 1)
            return key;
        return key + "_" + std::to_string(vpos - v.begin() + 1);
    }
    throw std::runtime_error("BUG: object not found in SampleLabelHandler");
}
