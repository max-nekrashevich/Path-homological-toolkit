from collections import deque
from copy import deepcopy


import numpy as np


import path_homology.graph as g
from path_homology import _params


def null_space(A, rcond=None):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q


def check_adjacency(adjacency: 'dict[g.Vertex, list[g.Vertex]]') -> bool:
    for neighbourhood in adjacency.values():
        for v in neighbourhood:
            if v not in adjacency:
                return False
    return True


def adjacency_from_matrix(adjacency_matrix: np.ndarray) -> 'dict[g.Vertex, list[g.Vertex]]':
    return {v: np.where(edge_to)[0].tolist() for v, edge_to in enumerate(adjacency_matrix)}


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


def compute_path_homology_dimension(graph: 'g.Graph', dim: int, regular: bool = False) -> int:
    graph = graph.prune()
    if _params['reduced']:
        return graph.get_dimH_n(dim, regular)
    subgraphs = graph.split()
    if dim == 0:
        return len(subgraphs)
    if not subgraphs:
        return 0
    return sum([subgraph.get_dimH_n(dim, regular) for subgraph in subgraphs])