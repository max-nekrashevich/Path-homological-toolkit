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
ListOfEdges = List[Tuple[Vertex, Vertex]]
Adjacency = Dict[Vertex, List[Vertex]]


class Graph(object):

    def __init__(self, adjacency: 'Adjacency | np.ndarray | ListOfEdges' = None) -> None:
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
        self._path_complex = p.GraphPathComplex(self)


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


    def from_epath(self, path: p.EPath, allowed: bool = False, order: int = ph.params.order) -> p.Path:
        return self._path_complex.from_epath(path, allowed, order)


    def get_d_matrix(self, n: int, *, allowed: bool = True,
                                      regular: bool = False,
                                      invariant: bool = False,
                                      order: int = None) -> np.ndarray:
        return self._path_complex.get_d_matrix(n, allowed=allowed, regular=regular, invariant=invariant, order=order)


    def get_A_n(self, dim: int, order: int = ph.params.order) -> List[p.Path]:
        return self._path_complex.get_A_n(dim, order)


    def get_Omega_n(self, dim: int, regular: bool = False, order: int = ph.params.order) -> List[p.Path]:
        return self._path_complex.get_Omega_n(dim, regular, order)


    def get_Z_n(self, dim: int, regular: bool = False, order: int = ph.params.order) -> List[p.Path]:
        return self._path_complex.get_Z_n(dim, regular, order)


    def get_dimH_n(self, dim: int, regular: bool = False, order: int = ph.params.order) -> int:
        return self._path_complex.get_dimH_n(dim, regular, order)