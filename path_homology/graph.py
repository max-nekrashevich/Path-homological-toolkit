from collections import Counter
from copy import deepcopy
from itertools import product, chain
from functools import lru_cache
from typing import Any, Dict, Iterable, List, NewType, Set, Tuple


import numpy as np


import path_homology as ph
import path_homology.path as p
import path_homology.utils as u


Vertex = NewType('Vertex', Any)
EPath = Tuple[Vertex, ...]
ListOfEdges = List[Tuple[Vertex, Vertex]]
Adjacency = Dict[Vertex, List[Vertex]]


class Graph(object):

    def __init__(self, adjacency: 'Adjacency | np.ndarray | ListOfEdges| None' = None) -> None:
        super().__init__()
        if adjacency is None:
            adjacency = {}
        if isinstance(adjacency, list):
            adjacency = u.adjacency_from_edges(adjacency)
        if isinstance(adjacency, np.ndarray):
            adjacency = u.adjacency_from_matrix(adjacency)

        if not u.check_adjacency(adjacency):
            raise Exception("Graph representation is invalid.")

        self._adjacency = adjacency
        self._update_attributes()


    def _update_attributes(self) -> None:
        self.n_vertices = len(self._adjacency)
        self._vertex_order = dict(zip(self._adjacency, range(len(self))))
        self.in_degrees = Counter(chain.from_iterable(self._adjacency.values()))
        self.out_degrees = {v: len(neighbors) for v, neighbors in self._adjacency.items()}


    def __len__(self) -> int:
        return self.n_vertices


    def copy(self) -> 'Graph':
        return Graph(deepcopy(self._adjacency))


    def add_vertex(self, to_add: Vertex) -> 'Graph':
        self._adjacency.setdefault(to_add, [])
        self._vertex_order.setdefault(to_add, self.n_vertices)
        self.in_degrees.setdefault(to_add, 0)
        self.out_degrees.setdefault(to_add, 0)
        self.n_vertices += 1
        return self


    def remove_vertices(self, to_remove: Iterable[Vertex]) -> 'Graph':
        self._adjacency = {v: [u for u in neighbors if u not in to_remove] for v, neighbors in self._adjacency.items() if v not in to_remove}
        self._update_attributes()
        return self


    def add_edge(self, start: Vertex, end: Vertex) -> 'Graph':
        if start not in self._adjacency:
            self.add_vertex(start)
        if end not in self._adjacency:
            self.add_vertex(end)
        if end not in self._adjacency[start]:
            self._adjacency[start].append(end)
        self.in_degrees[end] += 1
        self.out_degrees[start] += 1
        return self


    def remove_edge(self, start: Vertex, end: Vertex) -> 'Graph':
        if start in self._adjacency and end in self._adjacency[start]:
            self._adjacency[start].remove(end)
        self.in_degrees[end] -= 1
        self.out_degrees[start] -= 1
        return self


    def extend(self, other: 'Graph') -> 'Graph':
        for u in other._adjacency:
            for v in other._adjacency[u]:
                self.add_edge(u, v)
        return self


    def get_subgraph(self, vertices: Iterable[Vertex]) -> 'Graph':
        return Graph({v: [u for u in neighbors if u in vertices] for v, neighbors in self._adjacency.items() if v in vertices})


    def get_in_leaves(self) -> Set[Vertex]:
        return {v for v in self._adjacency if self.in_degrees[v] == 1 and self.out_degrees[v] == 0}


    def get_out_leaves(self) -> Set[Vertex]:
        return {v for v in self._adjacency if self.in_degrees[v] == 0 and self.out_degrees[v] == 1}


    def prune(self) -> 'Graph':
        graph = self
        while in_leaves := graph.get_in_leaves():
            graph = graph.copy()
            graph.remove_vertices(in_leaves)
            out_leaves = graph.get_out_leaves()
            if not out_leaves:
                break
            graph.remove_vertices(out_leaves)
        return graph


    def split(self):
        return [self.get_subgraph(component) for component in u.connected_components(u.to_undirected_graph(self._adjacency))]


    def list_paths(self, n: int, allowed: bool = True) -> 'list[EPath]':
        if allowed:
            return list(self._enum_allowed_paths(n))
        return list(self._enum_all_paths(n))


    @lru_cache(maxsize=10)
    def _enum_all_paths(self, n: int) -> Dict[EPath, int]:
        if n < 0 and not ph.params.reduced:
            return {}
        return {p: i for i, p in enumerate(product(self._adjacency, repeat=n+1))}


    @lru_cache(maxsize=20)
    def _enum_allowed_paths(self, n: int) -> Dict[EPath, int]:
        if n < 0:
            return {(): 0} if ph.params.reduced else {}
        if n == 0:
            return {(v, ) : i for v, i in self._vertex_order.items()}

        paths = self._enum_allowed_paths(n - 1)

        new_paths = {}
        i = 0
        for path in paths:
            for v in self._adjacency[path[-1]]:
                new_paths[path + (v,)] = i
                i += 1

        return new_paths


    def _clear_cache(self) -> None:
        self._enum_all_paths.cache_clear()
        self._enum_allowed_paths.cache_clear()
        self.get_d_matrix.cache_clear()


    def _path_index(self, path: EPath, allowed: bool = True) -> int:
        if allowed:
            return self._enum_allowed_paths(len(path) - 1)[path]
        return self._enum_all_paths(len(path) - 1)[path]


    def _get_coef_shape(self, n: int, allowed: bool) -> np.ndarray:
        if allowed:
            n_paths = len(self._enum_allowed_paths(n))
        else:
            n_paths = self.n_vertices ** (n + 1)
        return np.zeros(n_paths)


    def _non_allowed_ix(self, n: int) -> list:
        allowed = self._enum_allowed_paths(n)
        return [i for i, path in enumerate(self.list_paths(n, False)) if path not in allowed]


    def _allowed_ix(self, n: int) -> list:
        allowed = self._enum_allowed_paths(n)
        return [i for i, path in enumerate(self.list_paths(n, False)) if path in allowed]


    def from_epath(self, path: EPath, allowed: bool = False) -> p.Path:
        coefficients = self._get_coef_shape(len(path) - 1, allowed)
        coefficients[self._path_index(path, allowed)] = 1

        return p.Path(self, coefficients, len(path) - 1, allowed)


    @lru_cache(maxsize=10)
    def get_d_matrix(self, n: int, *, allowed: bool = True,
                                      regular: bool = False,
                                      invariant: bool = False) -> np.ndarray:
        paths = self.list_paths(n, allowed or invariant)
        if not ph.params.reduced and n == 0:
            return np.zeros((0, len(paths)))
        d: np.ndarray = np.zeros((self.n_vertices ** n, len(paths)))
        for i, path in enumerate(paths):
            for j in range(n + 1):
                if not regular or j == 0 or j == n or path[j - 1] != path[j + 1]:
                    d[self._path_index(path[:j] + path[j + 1:], False), i] += (-1) ** j
        return d[self._allowed_ix(n - 1)] if invariant else d


    def get_A_n(self, dim: int) -> List[p.Path]:
        return [self.from_epath(path) for path in self.list_paths(dim, True)]

    def get_Omega_n(self, dim: int, regular: bool = False) -> List[p.Path]:
        constraints = self.get_d_matrix(dim, regular=regular)[self._non_allowed_ix(dim - 1)]
        if constraints.shape[0] == 0:
            return self.get_A_n(dim)
        if constraints.shape[1] == 0:
            return []
        weights = u.null_space(constraints).T
        return [p.Path(self, weight, dim, True, True) for weight in weights]


    def get_Z_n(self, dim: int, regular: bool = False) -> List[p.Path]:
        constraints = self.get_d_matrix(dim, regular=regular)
        if constraints.shape[0] == 0:
            return self.get_A_n(dim)
        if constraints.shape[1] == 0:
            return []
        weights = u.null_space(constraints).T
        return [p.Path(self, weight, dim, True, True) for weight in weights]


    def get_dimH_n(self, dim: int, regular: bool = False) -> int:
        dim_Z_n: int = len(self.get_Z_n(dim, regular))
        B_n = [path.d(regular).coefficients for path in self.get_Omega_n(dim + 1, regular)]
        dim_B_n = np.linalg.matrix_rank(np.stack(B_n)) if B_n else 0
        return dim_Z_n - dim_B_n
