import numpy as np
import vtk
from paraview.util.vtkAlgorithm import (VTKPythonAlgorithmBase, smdomain,
                                        smproperty, smproxy)
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkImageData


@smproxy.source(label="Sine Wave Image Source")
class SineWaveImageSource(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkImageData")
        self.grid_size = 10
        self.amplitude = 1.0

    @smproperty.intvector(name="GridSize", default_values=10)
    @smdomain.intrange(min=1, max=100)
    def SetGridSize(self, grid_size):
        if self.grid_size != grid_size:
            self.grid_size = grid_size
            self.Modified()

    @smproperty.doublevector(name="Amplitude", default_values=1.0)
    @smdomain.doublerange(min=0.1, max=10.0)
    def SetAmplitude(self, amplitude):
        if self.amplitude != amplitude:
            self.amplitude = amplitude
            self.Modified()

    def RequestInformation(self, request, inInfo, outInfo):
        executive = self.GetExecutive()
        in_info = inInfo[0].GetInformationObject(0)
        if not in_info.Has(executive.WHOLE_EXTENT()):
            return 1

        extent = list(in_info.Get(executive.WHOLE_EXTENT()))
        dims = [extent[2 * i + 1] - extent[2 * i] + 1 for i in range(len(extent) // 2)]

        out_info = outInfo.GetInformationObject(0)
        out_info.Set(executive.WHOLE_EXTENT(), extent, 6)
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        executive = self.GetExecutive()
        in_info = inInfo[0].GetInformationObject(0)
        in_info.Set(executive.UPDATE_EXTENT(), in_info.Get(executive.WHOLE_EXTENT()), 6)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        # Get the output data object
        input = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output = dsa.WrapDataObject(vtkImageData.GetData(outInfo, 0))
        
        #  -- Do the computation here --

        # Copy the input to the output as example
        output.ShallowCopy(input.VTKObject)

        return 1
