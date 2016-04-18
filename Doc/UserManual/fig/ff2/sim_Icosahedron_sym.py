"""
Plot form factors.
"""
import bornagain as ba
from   bornagain import nanometer, degree
import bornplot as bp
import math

det = bp.Detector( 200, -5, 5, -5, 5 )
n    = 3
results = []
edge = 4.8

title = 'face normal'
trafo = ba.RotationY(48.1897*degree)
ff = ba.FormFactorIcosahedron(edge*nanometer)
data = bp.run_simulation(det,ff,trafo)
results.append( bp.Result(0, data, title) )

title = 'vertex normal'
trafo = ba.RotationY(-52.6226*degree)
ff = ba.FormFactorIcosahedron(edge*nanometer)
data = bp.run_simulation(det,ff,trafo)
results.append( bp.Result(1, data, title) )

title = 'edge normal'
trafo = ba.RotationY(69.0948*degree)
ff = ba.FormFactorIcosahedron(edge*nanometer)
data = bp.run_simulation(det,ff,trafo)
results.append( bp.Result(2, data, title) )

bp.make_plot( results, det, "ff_Icosahedron_sym" )
