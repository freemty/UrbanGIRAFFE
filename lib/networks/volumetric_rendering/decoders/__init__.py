from .z_grid import SPADEGenerator3D, SPADEGenerator2D
from .eg3d import EG3DDecoder as eg3dDecoder
from .giraffe import giraffeDecoder, giraffeInsDecoder
# from .style import NerfStyleDecoder as styleDecoder
# from .style import LocalFeatureDecoder as localfeatureDecoder
from .style import stuffDecoder, StyleDecoder2D

# from .z_map import SPADEGenerator, simpleGenerator
from .eg3d import TriplaneGenerator as triplaneGenerator
from .sky import skyDecoder
from .cnn import CNNRender as gancraftNR
# from .cnn import NeuralRenderer as giraffeNR
