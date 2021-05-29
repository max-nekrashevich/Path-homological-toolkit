from dataclasses import dataclass
from functools import lru_cache
from itertools import cycle, product
from typing import Dict, List, Tuple
from warnings import simplefilter


import numpy as np


from .types import EPath, Vertex
import path_homology.graph as g
import path_homology.utils as u


@dataclass()
class Path(object):

    _path_complex: 'BasePathComplex'
    coefficients: 'np.ndarray'
    length: int
    order: int
    allowed: bool
    invariant: bool = False


    @staticmethod
    def _format_term(term: Tuple[bool, float, EPath], first: bool = False) -> str:
        sign, coefficient, path = term

        if first:
            formatted_term = "" if sign else "-"
        else:
            formatted_term = " + " if sign else " - "
        if abs(coefficient - 1) > u.params.eps:
            formatted_term += str(round(coefficient, u.params.n_decimal))
        formatted_term += u.params.epath_outer.format(u.params.epath_delim.join(map(str, path)))
        return formatted_term


    def _get_terms(self) -> List[Tuple[bool, float, EPath]]:
        terms = []
        paths = self._path_complex.get_paths(self.length, self.allowed)

        for coef, path in zip(self.coefficients, paths): # type: ignore
            if abs(coef) > u.params.eps:
                terms.append((coef > 0, abs(coef), path))

        return terms


    def __str__(self) -> str:
        terms = self._get_terms()

        if not terms:
            return "0"

        path_string: str = ""
        for i, term in enumerate(terms):
            path_string += self._format_term(term, i == 0)

        return path_string


    def __repr__(self) -> str:
        if not u.params.raw_repr:
            return str(self)
        return f"{self.length}-Path({str(self.coefficients)}, graph_id={hex(id(self._path_complex))}, allowed={self.allowed}, invariant={self.invariant})"


    def __add__(self, other: 'Path') -> 'Path':
        if self._path_complex != other._path_complex:
            raise ArithmeticError("Paths are from different graphs")
        if self.length != other.length:
            raise ArithmeticError("Paths have different lengths")

        if self.allowed != other.allowed:
            path1 = self.to_non_allowed()
            path2 = other.to_non_allowed()
            return Path(self._path_complex, path1.coefficients - path2.coefficients, self.length, self.order, False)
        return Path(self._path_complex, self.coefficients + other.coefficients, self.length, self.order, self.allowed)


    def __mul__(self, c: 'int | float') -> 'Path':
        return Path(self._path_complex, c * self.coefficients, self.length, self.order, self.allowed, self.invariant)


    __rmul__ = __mul__


    def __truediv__(self, c: 'int | float') -> 'Path':
        return Path(self._path_complex, self.coefficients / c, self.length, self.order, self.allowed, self.invariant) # type: ignore


    def __sub__(self, other: 'Path') -> 'Path':
        return self + (other * -1)


    def to_non_allowed(self) -> 'Path':
        if not self.allowed:
            return self

        new_coefficients = self._path_complex._get_coef_shape(self.length, False, self.order)
        paths = self._path_complex.get_paths(self.length, self.allowed)
        for coef, path in zip(self.coefficients, paths): # type: ignore
            new_coefficients[self._path_complex._path_index(path, False)] = coef
        return Path(self._path_complex, new_coefficients, self.length, self.order, False)


    def is_allowed(self) -> bool:
        if self.allowed:
            return True

        paths = list(self._path_complex.get_all_paths(self.length))
        for i in self._path_complex._non_allowed_ix(self.length):
            if abs(self.coefficients[self._path_complex._path_index(paths[i], self.allowed)]) > u.params.eps:
                return False
        return True


    def to_allowed(self, restrict=False) -> 'Path':
        assert restrict or self.is_allowed(), "Has non-zero non-allowed terms and can't be converted."
        ix = [self._path_complex._path_index(path, self.allowed) for path in self._path_complex.get_allowed_paths(self.length)]
        return Path(self._path_complex, self.coefficients[ix], self.length, self.order, True)


    def is_invariant(self) -> bool:
        if self.invariant:
            return True
        return self.d().is_allowed()


    def d(self, regular: bool = False) -> 'Path':
        new_coefficients = self._path_complex.get_d_matrix(self.length, allowed=self.allowed,
                                                             regular=regular,
                                                             invariant=self.invariant,
                                                             order=self.order) @ self.coefficients
        return Path(self._path_complex, new_coefficients, self.length - 1, self.order, self.invariant, self.invariant)



class BasePathComplex(object):

    def get_all_paths(self, n: int) -> Dict[EPath, int]:
        raise NotImplementedError("BasePathComplex is an abstract class")


    def get_allowed_paths(self, n: int) -> Dict[EPath, int]:
        raise NotImplementedError("BasePathComplex is an abstract class")


    def get_paths(self, n: int, allowed: bool) -> Dict[EPath, int]:
        return self.get_allowed_paths(n) if allowed else self.get_all_paths(n)


    def get_vertices(self) -> Dict[Vertex, int]:
        raise NotImplementedError("BasePathComplex is an abstract class")


    def n_vertices(self) -> int:
        raise NotImplementedError("BasePathComplex is an abstract class")

    def _path_index(self, path: EPath, allowed: bool = True) -> int:
        if allowed:
            return self.get_allowed_paths(len(path) - 1)[path]
        return self.get_all_paths(len(path) - 1)[path]


    def _get_coef_shape(self, n: int, allowed: bool, order: int = 0) -> np.ndarray:
        if allowed:
            n_paths = len(self.get_allowed_paths(n))
        else:
            n_paths = self.n_vertices() ** (n + 1)
        return u.zeros(n_paths, order)


    def _non_allowed_ix(self, n: int) -> list:
        allowed = self.get_allowed_paths(n)
        return [i for i, path in enumerate(self.get_all_paths(n)) if path not in allowed]


    def _allowed_ix(self, n: int) -> list:
        allowed = self.get_allowed_paths(n)
        return [i for i, path in enumerate(self.get_all_paths(n)) if path in allowed]


    def from_epath(self, path: EPath, allowed: bool = False, order: int = 0) -> Path:
        coefficients = self._get_coef_shape(len(path) - 1, allowed, order)
        coefficients[self._path_index(path, allowed)] = 1

        return Path(self, coefficients, len(path) - 1, order, allowed)


    @lru_cache(maxsize=10)
    def get_d_matrix(self, n: int, *, allowed: bool = True,
                                      regular: bool = False,
                                      invariant: bool = False,
                                      order: int = 0) -> np.ndarray:
        paths = self.get_paths(n, allowed or invariant)
        if n == 0:
            return u.zeros((0, len(paths)), order)
        d: np.ndarray = u.zeros((self.n_vertices() ** n, len(paths)), order)
        signs = u.get_signs(order)
        for i, path in enumerate(paths):
            for j, coef in zip(range(n + 1), cycle(signs)):
                if not regular or j == 0 or j == n or path[j - 1] != path[j + 1]:
                    d[self._path_index(path[:j] + path[j + 1:], False), i] += coef
        return d[self._allowed_ix(n - 1)] if invariant else d


    def get_A_n(self, dim: int, order: int = 0) -> List[Path]:
        return [self.from_epath(path, order=order) for path in self.get_allowed_paths(dim)]


    def get_Omega_n(self, dim: int, regular: bool = False, order: int = 0) -> List[Path]:
        constraints = self.get_d_matrix(dim, regular=regular, order=order)[self._non_allowed_ix(dim - 1)]
        if constraints.shape[0] == 0:
            return self.get_A_n(dim, order)
        if constraints.shape[1] == 0:
            return []
        weights = u.null_space(constraints, order).T
        return [Path(self, weight, dim, order, True, True) for weight in weights]


    def get_Z_n(self, dim: int, regular: bool = False, order: int = 0) -> List[Path]:
        constraints = self.get_d_matrix(dim, regular=regular, order=order)
        if constraints.shape[0] == 0:
            return self.get_A_n(dim, order)
        if constraints.shape[1] == 0:
            return []
        weights = u.null_space(constraints, order).T
        return [Path(self, weight, dim, order, True, True) for weight in weights]


    def get_dimH_n(self, dim: int, regular: bool = False, order: int = 0) -> int:
        dim_Z_n: int = len(self.get_Z_n(dim, regular, order))
        B_n = [path.d(regular).coefficients for path in self.get_Omega_n(dim + 1, regular, order)]
        dim_B_n = np.linalg.matrix_rank(np.stack(B_n)) if B_n else 0
        return dim_Z_n - dim_B_n


    def _clear_cache(self) -> None:
        self.get_d_matrix.cache_clear()


@dataclass(frozen=True)
class GraphPathComplex(BasePathComplex):
    _graph: 'g.Graph'

    @lru_cache(maxsize=10)
    def get_all_paths(self, n: int) -> Dict[EPath, int]:
        if n < 0:
            return {}
        return {p: i for i, p in enumerate(product(self._graph._adjacency, repeat=n+1))}


    @lru_cache(maxsize=20)
    def get_allowed_paths(self, n: int) -> Dict[EPath, int]:
        if n < 0:
            return {}
        if n == 0:
            return {(v, ) : i for v, i in self.get_vertices().items()}

        paths = self.get_allowed_paths(n - 1)

        new_paths = {}
        i = 0
        for path in paths:
            for v in self._graph._adjacency[path[-1]]:
                new_paths[path + (v,)] = i
                i += 1

        return new_paths


    def get_vertices(self) -> Dict[Vertex, int]:
        return self._graph._vertex_order


    def n_vertices(self) -> int:
        return len(self._graph._adjacency)


    def _clear_cache(self) -> None:
        self.get_all_paths.cache_clear()
        self.get_allowed_paths.cache_clear()
        self.get_d_matrix.cache_clear()


class PathComplex(BasePathComplex):
    pass
