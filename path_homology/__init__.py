_params = dict(eps=1e-5,
               output_format=dict(raw_repr=False,
                                n_decimal=2,
                                epath_outer='({})',
                                epath_delim='→'),
               reduced=False)

__version__ = "0.1.2"

from .graph import Graph
from .utils import compute_path_homology_dimension



