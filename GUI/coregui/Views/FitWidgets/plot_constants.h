// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/FitWidgets/plot_constants.h
//! @brief     Defines various constants for plotting
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef PLOT_CONSTANTS_H
#define PLOT_CONSTANTS_H

#include "StyleUtils.h"
#include <QSize>

namespace Constants
{

inline int plot_tick_label_size()
{
    return StyleUtils::SystemPointSize() * 0.9;
}

inline int plot_axes_label_size()
{
    return StyleUtils::SystemPointSize();
}

inline int plot_colorbar_size()
{
    return StyleUtils::SizeOfLetterM().width();
}

} // namespace Constants

#endif // PLOT_CONSTANTS_H
