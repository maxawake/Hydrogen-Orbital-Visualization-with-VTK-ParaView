import vtk
import numpy as np
from typing import Union

from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import (vtkStructuredGrid, vtkDataSet, VTK_VERTEX)
from vtkmodules.numpy_interface import dataset_adapter as dsa

# Website https://gitlab.kitware.com/paraview/paraview/blob/master/Utilities/Doxygen/pages/PluginHowto.md

from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

from paraview.util import SetOutputWholeExtent

@smproxy.source(name="VectorFieldGenerating",label="Generate Vectorfield")
class VectorFieldGenerating(VTKPythonAlgorithmBase):
    def __init__(self) -> None:
        # Initialization of the super class with data output vtkImageData
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1, outputType='vtkImageData')
        
        self.vtkImageData = vtk.vtkImageData()
        
    # Don't really know what it dos, but it is essential
    def RequestInformation(self, request, inInfoVec, outInfoVec):
        dims = self.vtkImageData.GetDimensions()
        SetOutputWholeExtent(self,[0, dims[0]- 1, 0, dims[1] - 1, 0, dims[2] - 1])
        return 1    
        
    def RequestData(self, request, inputs, outputData):     
        # Definition of the output file and its datatype
        output = dsa.WrapDataObject(vtk.vtkImageData.GetData(outputData, 0)) 
        
        # Copy the vtkVectorfield2DImage_Ground Object to output 
        output.ShallowCopy(self.vtkImageData) 
        return 1
    
    @smproperty.xml("""<IntVectorProperty
                    name="Dimension"
                    command="SetDimension"
                    number_of_elements="3"
                    default_values="64 64 64">
                    </IntVectorProperty>""")
    def SetDimension(self, x: int, y: int, z: int) -> None:
        """Set dimensions for the structured points grid
        
        : param x: dimension of the x-axis
        : param y: dimension of the y-axis
        : param z: dimension of the z-axis
        """
        self.vtkImageData.SetDimensions(x, y, z)
        self.Modified()
        
        
    @smproperty.xml("""<DoubleVectorProperty
                    name="Spacing"
                    command="SetSpacing"
                    number_of_elements="3"
                    default_values="1.0 1.0 1.0">
                    </DoubleVectorProperty>""")
    def SetSpacing(self, x: float, y: float, z: float) -> None:
        """Set spacing for the structured points grid
        
        : param x: spacing of the x-axis
        : param y: spacing of the y-axis
        : param z: spacing of the z-axis
        """
        self.vtkImageData.SetSpacing(x, y, z)
        self.Modified()
        
