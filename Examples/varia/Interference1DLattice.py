"""
Simulation of grating using very long boxes and 1D lattice.
Monte-carlo integration is used to get rid of
large-particle form factor oscillations.
"""
import bornagain as ba
from bornagain import angstrom, deg, nm, nm2, kvector_t


def get_sample():
    """
    Returns a sample with a grating on a substrate,
    modelled by very long boxes forming a 1D lattice with Cauchy correlations.
    """

    # Define materials
    material_Particle = ba.HomogeneousMaterial("Particle", 0.0006, 2e-08)
    material_Substrate = ba.HomogeneousMaterial("Substrate", 6e-06, 2e-08)
    material_Vacuum = ba.HomogeneousMaterial("Vacuum", 0.0, 0.0)

    # Define form factors
    ff = ba.FormFactorBox(10.0*nm, 10000.0*nm, 10.0*nm)

    # Define particles
    particle = ba.Particle(material_Particle, ff)
    particle_rotation = ba.RotationZ(45.0*deg)
    particle.setRotation(particle_rotation)

    # Define interference functions
    iff = ba.InterferenceFunction1DLattice(30.0*nm, 45.0*deg)
    iff_pdf = ba.FTDecayFunction1DCauchy(1000.0*nm)
    iff.setDecayFunction(iff_pdf)

    # Define particle layouts
    layout = ba.ParticleLayout()
    layout.addParticle(particle, 1.0)
    layout.setInterferenceFunction(iff)
    layout.setWeight(1)
    layout.setTotalParticleSurfaceDensity(0.01)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_1.addLayout(layout)
    layer_2 = ba.Layer(material_Substrate)

    # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)

    return sample


def get_simulation(sample):
    beam = ba.Beam(1.0, 1.0*angstrom, ba.Direction(0.2*deg, 0.0*deg))
    det = ba.SphericalDetector(200, -1*deg, 1*deg, 200, 0*deg, 2*deg)
    simulation = ba.GISASSimulation(beam, sample, det)
    simulation.getOptions().setMonteCarloIntegration(True, 100)
    if not "__no_terminal__" in globals():
        simulation.setTerminalProgressMonitor()
    return simulation


if __name__ == '__main__':
    ba.run_and_plot(get_simulation(get_sample()), intensity_min=1e-03)
