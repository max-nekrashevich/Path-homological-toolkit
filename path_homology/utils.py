from collections import deque, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple


import numpy as np

try:
    import galois as gl

except ImportError:
    print("WARNING: Galois package not installed.")
    print("         This package requires galois for finite field computations.")
    print("         To install it run: $ pip install galois")


from .types import *


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


params = Params()


def get_signs(order: int) -> tuple:
    if order == 0:
        return (1, -1)
    else:
        return (gl.GF(order)(1), gl.GF(order)(order - 1))


def zeros(shape: 'int | Tuple[int, ...]', order: int) -> np.ndarray:
    if order == 0:
        return np.zeros(shape, int)
    else:
        return gl.GF(order).Zeros(shape) # type: ignore


def null_space(A, order):
    if order == 0:
        return null_space_numpy(A)
    else:
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
    ext_A = np.vstack([A, GF(order).Identity(N)]).T # type: ignore
    red_A = ext_A.row_reduce(M)
    num = np.sum(np.any(red_A[:N, :M] != 0, axis=1))
    return red_A[num:, M:].T


def check_adjacency(adjacency: Adjacency) -> bool:
    for neighbourhood in adjacency.values():
        for v in neighbourhood:
            if v not in adjacency:
                return False
    return True


def adjacency_from_matrix(adjacency_matrix: np.ndarray) -> Adjacency:
    return {v: np.where(edge_to)[0].tolist() for v, edge_to in enumerate(adjacency_matrix)} # type: ignore


def adjacency_from_edges(list_of_edjes: ListOfEdges) -> Adjacency:
    adjacency = defaultdict(list)
    for start, end in list_of_edjes:
        adjacency[start].append(end)
        adjacency[end]
    return adjacency


def to_undirected_graph(adjacency: Adjacency) -> Adjacency:
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
