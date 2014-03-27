#include "ProgressHandler.h"
#include "Exceptions.h"
#include "Simulation.h"
#include <boost/thread.hpp>


ProgressHandler::ProgressHandler()
    : m_nitems(0)
    , m_nitems_max(0)
    , m_current_progress(0)
{

}

void ProgressHandler::reset()
{
    m_nitems = 0;
    m_nitems_max = 0;
    m_current_progress = 0;
    m_callback = NULL;
}


//! Collects number of items processed by different DWBASimulation's.
//! Calculates general progress and inform GUI if progress has changed.
//! Return flag is obtained from GUI and transferred to DWBASimulation to ask
//! them to stop calculations.
bool ProgressHandler::update(int n)
{
    static boost::mutex single_mutex;
    boost::unique_lock<boost::mutex> single_lock( single_mutex );

    // this flag is to inform Simulation that GUI wants it to be terminated
    bool continue_calculations(true);

    m_nitems += n;

    std::cout << "ProgressHandler::update " << m_nitems << std::endl;
    int progress = int(double(100*m_nitems)/double(m_nitems_max)); // in percents
    if(progress != m_current_progress) {
        m_current_progress = progress;
        if(m_callback) {
            continue_calculations = m_callback(m_current_progress); // report to gui
        }
    }
    return continue_calculations;
}


//! Initialize ProgressHandler, estimates number of items to be calculated
//! by DWBASimulation's.
void ProgressHandler::init(Simulation *simulation, int param_combinations)
{
    m_nitems = 0;
    m_current_progress = 0;
    m_nitems_max = 0;

    // Here we could run through the multilayer to define number of DecoratedDWBASimulation's
    // for precise estimation of number of items to be processed.

    // Simplified estimation of total number of items in DWBA simulation
    m_nitems_max = param_combinations*simulation->getOutputData()->getAllocatedSize();

    //m_nitems_max *= 2; //diffuse and non diffuse case

    std::cout << "ProgressHandler::init() -> m_nitems_max" << m_nitems_max << std::endl;
}