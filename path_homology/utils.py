from collections import deque, defaultdict
from copy import deepcopy
from dataclasses import dataclass


import galois as gl
import numpy as np


import path_homology as ph
import path_homology.graph as g


@dataclass()
class Params:

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Params, cls).__new__(cls)
        return cls.instance

    eps: float = 1e-5
    n_decimal: int = 2
    epath_outer: str = '({})'
    epath_delim: str = '→'
    raw_repr: bool = False
    reduced: bool = False
    order: int = 0


def null_space(A, order):
    if order == 0:
        return null_space_numpy(A)
    return null_space_galois(A, order)


def null_space_numpy(A):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q


def null_space_galois(A, order):
    M, N = A.shape
    ext_A = np.vstack([A, gl.GF(order).Identity(N)]).T # type: ignore
    return ext_A.row_reduce(M)[M:, M:].T


def check_adjacency(adjacency: g.Adjacency) -> bool:
    for neighbourhood in adjacency.values():
        for v in neighbourhood:
            if v not in adjacency:
                return False
    return True


def adjacency_from_matrix(adjacency_matrix: np.ndarray) -> g.Adjacency:
    return {v: np.where(edge_to)[0].tolist() for v, edge_to in enumerate(adjacency_matrix)} # type: ignore


def adjacency_from_edges(list_of_edjes: g.ListOfEdges) -> g.Adjacency:
    adjacency = defaultdict(list)
    for start, end in list_of_edjes:
        adjacency[start].append(end)
        adjacency[end]
    return adjacency


def to_undirected_graph(adjacency):
    graph = deepcopy(adjacency)
    for v, neighbors in adjacency.items():
        for u in neighbors:
            graph[u].append(v)
    return graph


def connected_components(undirected_graph):
    seen = set()

    for root in undirected_graph:
        if root not in seen:
            seen.add(root)
            component = set()
            queue = deque([root])

            while queue:
                node = queue.popleft()
                component.add(node)
                for neighbor in undirected_graph[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            yield component


def compute_path_homology_dimension(graph: g.Graph, dim: int, regular: bool = False) -> int:
    graph = graph.prune()
    if ph.params.reduced:
        return graph.get_dimH_n(dim, regular)
    subgraphs = graph.split()
    if dim == 0:
        return len(subgraphs)
    if not subgraphs:
        return 0
    return sum([subgraph.get_dimH_n(dim, regular) for subgraph in subgraphs])


def Cycle(n: int, k: int = 1, vertex_labels: 'list[g.Vertex] | None' = None) -> g.Graph:
    if vertex_labels is None:
        vertex_labels = list(range(n))
    adjacency = {vertex_labels[i]: [vertex_labels[(i + 1 + j) % n] for j in range(k)] for i in range(n)}
    return g.Graph(adjacency)


def Simplex(n: int, vertex_labels: 'list[g.Vertex] | None' = None) -> g.Graph:
    if vertex_labels is None:
        vertex_labels = list(range(n))
    adjacency = {vertex_labels[i]: [vertex_labels[j] for j in range(i + 1, n)] for i in range(n)}
    return g.Graph(adjacency)
