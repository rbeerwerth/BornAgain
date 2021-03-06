'''
Simulation demo: Cylinder and/or sphere on substrate
'''

import numpy
import matplotlib
import math
from bornagain import *

# ----------------------------------
# describe sample and run simulation
# ----------------------------------
def RunSimulation():
    # defining materials
    mAir = HomogeneousMaterial("Air", 0.0, 0.0)
    mSubstrate = HomogeneousMaterial("Substrate", 6e-6, 2e-8)
    mParticle = HomogeneousMaterial("Particle", 6e-4, 2e-8)

    # collection of particles
    cylinder_ff = FormFactorCylinder(2 * nanometer, 5 * nanometer)
    cylinder = Particle(mParticle, cylinder_ff)
    sphere_ff = FormFactorFullSphere(4 * nanometer)
    sphere = Particle(mParticle, sphere_ff)
    particle_layout = ParticleLayout()
    particle_layout.addParticle(cylinder)
    particle_layout.addParticle(sphere)
    interference = InterferenceFunctionRadialParaCrystal(20 * nanometer)
    pdf = FTDistribution1DGauss(2 * nanometer)
    interference.setProbabilityDistribution(pdf)
    particle_layout.setInterferenceFunction(interference)

    # air layer with particles and substrate form multi layer
    air_layer = Layer(mAir)
    air_layer.addLayout(particle_layout)
    substrate_layer = Layer(mSubstrate)
    multi_layer = MultiLayer()
    multi_layer.addLayer(air_layer)
    multi_layer.addLayer(substrate_layer)

    # build and run experiment
    simulation = GISASSimulation()
    simulation.setDetectorParameters(100, -4.0 * degree, 4.0 * degree, 100, 0.0 * degree, 8.0 * degree)
    simulation.setBeamParameters(1.0 * angstrom, 0.2 * degree, 0.0 * degree)
    simulation.setSample(multi_layer)
    simulation.runSimulation()
    # intensity data
    return simulation.result().array()


#-------------------------------------------------------------
# main()
#-------------------------------------------------------------
if __name__ == '__main__':
    result = RunSimulation() + 1 # for log scale
    im = plt.imshow(result,
                 norm=matplotlib.colors.LogNorm(),
                 extent=[-4.0, 4.0, 0, 8.0])
    plt.colorbar(im)
    plt.xlabel(r'$\phi_f$', fontsize=20)
    plt.ylabel(r'$\alpha_f$', fontsize=20)
    plt.show()
