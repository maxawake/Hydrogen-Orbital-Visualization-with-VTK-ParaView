import sys

sys.path.append("/home/max/Nextcloud/Physik/Computations/HydroVis")
import numpy as np
from hydrogen import *
from paraview.util.vtkAlgorithm import smdomain, smproperty, smproxy
from vtkModelBase import *


@smproxy.source(label="Hydrogen Atom Orbitals")
@vtk_model(dimensions=[64, 64, 64], extent=[-30, 30, -30, 30, -30, 30])
class vtkHydrogen(vtkModelBase):
    def __init__(self):
        vtkModelBase.__init__(self)
        self._time = 0
        self._n = 2
        self._l = 1
        self._m = 0

    @smproperty.doublevector(name="TimeValue", label="Time Value", default_values=0.0)
    @smdomain.doublerange(min=0.0, max=10.0)
    def SetTimeValue(self, t):
        self._time = t
        self.Modified()

    @smproperty.intvector(name="QuantumNumbers", label="Quantum Numbers", default_values=[3, 2, 0])
    def SetQuantumNumbers(self, n, l, m):
        self._n = n
        self._l = l
        self._m = m
        self.Modified()

    @smproperty.intvector(name="Superposition", label="Superposition", default_values=0)
    @smdomain.boolean()
    def superposition(self, superpos):
        self._superpos = superpos
        self.Modified()

    @smproperty.intvector(name="QuantumNumbers2", label="Quantum Numbers 2", default_values=[2, 1, 0])
    def SetQuantumNumbers2(self, n, l, m):
        self._n2 = n
        self._l2 = l
        self._m2 = m
        self.Modified()

    def Sample(self, x, y, z):
        # Convert the cartesian coordinates to spherical coordinates
        r, theta, phi = cartesian_to_spherical(x * a_0, y * a_0, z * a_0)

        # Calculate the wavefunction of the hydrogen atom at the given point
        psi1 = hydrogen_wavefunction(self._n, self._l, self._m, r, theta, phi)

        if self._superpos:
            psi2 = hydrogen_wavefunction(self._n2, self._l2, self._m2, r, theta, phi)
            
            # Calculate the energy of the two states
            E_n1 = eigenenergies(self._n)
            E_n2 = eigenenergies(self._n2)

            # Calculate the superposition of the two time-dependent wavefunctions
            psi = psi1 * np.exp(-1j * E_n1 * self._time) + psi2 * np.exp(-1j * E_n2 * self._time)
        else:
            psi = psi1

        # Calculate the probability density of the wavefunction at the given point as the square of the absolute value of the wavefunction
        dx = np.abs(psi) ** 2
        dy = np.abs(psi) ** 2
        dz = np.abs(psi) ** 2

        return np.stack([dx, dy, dz], axis=-1)
