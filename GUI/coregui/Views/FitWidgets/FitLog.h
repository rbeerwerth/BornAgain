// ************************************************************************** //
//
//  BornAgain: simulate and fit scattering at grazing incidence
//
//! @file      GUI/coregui/Views/FitWidgets/FitLog.h
//! @brief     Defines class FitLog
//!
//! @homepage  http://www.bornagainproject.org
//! @license   GNU General Public License v3 or higher (see COPYING)
//! @copyright Forschungszentrum Jülich GmbH 2018
//! @authors   Scientific Computing Group at MLZ (see CITATION, AUTHORS)
//
// ************************************************************************** //

#ifndef FITLOG_H
#define FITLOG_H

#include "FitLogFlags.h"
#include "WinDllMacros.h"
#include <string>
#include <vector>

class JobMessagePanel;

class BA_CORE_API_ FitLog
{
public:
    FitLog();

    void setMessagePanel(JobMessagePanel* messagePanel);
    void append(const std::string& text, FitLogFlags::MessageType type);
    void clearLog();

private:
    struct Record {
        std::string m_text;
        FitLogFlags::MessageType m_type;
    };

    std::vector<Record> m_records;
    JobMessagePanel* m_messagePanel;
};

#endif
