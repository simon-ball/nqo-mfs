#from .sources import Magnet, CircularCoil, RectangularCoil, PermanentMagnet, CoilPair
#
#from .helpers import rotate_around_z, rotate_around_x, rotate_to_dashed_frame, rotate_to_normal_frame
#from .helpers import evaluate_axis_projection, plot_scalar_B_field, plot_vector_B_field_threaded, print_field_gradient

from . import helpers
from . import sources
from . import examples

__version__ = "1.0.0"

#__all__ = ("Magnet", "CircularCoil", "RectangularCoil", "PermanentMagnet", "CoilPair", 
#           "rotate_around_z", "rotate_around_x", "rotate_to_dashed_frame", "rotate_to_normal_frame",
#           "evaluate_axis_projection", "plot_scalar_B_field", "plot_scalar_B_field_threaded", "print_field_gradient")