import sys
sys.path.append("/home/max/Nextcloud/Physik/Computations/HydroVis")
from vtkModelBase import *
import numpy as np
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain


@smproxy.source(label="VTK Model")
@vtk_model(dimensions=[128, 128, 128], extent=[-5, 5, -5, 5, -5, 5])
class vtkModel(vtkModelBase):
    def __init__(self):
        vtkModelBase.__init__(self)
        self._time = 0

    @smproperty.doublevector(name="TimeValue", label="Time Value", default_values=0.0)
    @smdomain.doublerange(min=0.0, max=10.0)
    def SetTimeValue(self, t):
        self._time = t
        self.Modified()

    def Sample(self, x, y, z):
        dx = self._time*x**2
        dy = self._time*y**2
        dz = self._time*z**2

        return np.stack([dx, dy, dz], axis=-1)
