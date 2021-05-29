import os


__version__ = None


FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.txt")
if os.path.exists(FILE):
    with open(FILE) as f:
        __version__ = f.read().split("\n")[0]


from .utils import params
from .graph import Graph, Cycle, Simplex


def compute_path_homology_dimension(graph: Graph, dim: int, regular: bool = False, order: int = 0) -> int:
    graph = graph.prune()
    subgraphs = graph.split()
    if dim == 0:
        return len(subgraphs)
    if not subgraphs:
        return 0
    return sum([subgraph.get_dimH_n(dim, regular, order) for subgraph in subgraphs])
