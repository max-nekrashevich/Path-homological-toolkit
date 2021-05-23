from dataclasses import dataclass


import numpy as np


import path_homology.graph as g
from path_homology import _params


@dataclass(frozen=True)
class Path(object):

    _graph: 'g.Graph'
    coefficients: np.ndarray
    length: int
    allowed: bool
    invariant: bool = False

    @staticmethod
    def _format_term(term: 'tuple[bool, float, g.EPath]', first: bool = False) -> str:
        sign, coefficient, path = term

        if first:
            formatted_term = "" if sign else "-"
        else:
            formatted_term = " + " if sign else " - "
        if abs(coefficient - 1) > _params['eps']:
            formatted_term += str(round(coefficient, _params['output_format']['n_decimal']))
        formatted_term += _params['output_format']['epath_outer'].format(_params['output_format']['epath_delim'].join(map(str, path)))
        return formatted_term

    def _get_terms(self) -> 'list[tuple[bool, float, g.EPath]]':
        terms = []
        paths = self._graph.list_paths(self.length, self.allowed)

        for coef, path in zip(self.coefficients, paths): # type: ignore
            if abs(coef) > _params['eps']:
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
        if not _params['output_format']['raw_repr']:
            return str(self)
        return f"{self.length}-Path({str(self.coefficients)}, graph_id={hex(id(self._graph))}, allowed={self.allowed}, invariant={self.invariant})"

    def __add__(self, other: 'Path') -> 'Path':
        assert self._graph == other._graph, "Paths are from different graphs"
        assert self.length == other.length, "Paths have different lengths"
        if self.allowed != other.allowed:
            path1 = self.to_non_allowed()
            path2 = other.to_non_allowed()
            return Path(self._graph, path1.coefficients - path2.coefficients, self.length, False)
        return Path(self._graph, self.coefficients + other.coefficients, self.length, self.allowed)

    def __mul__(self, c: 'int | float') -> 'Path':
        return Path(self._graph, c * self.coefficients, self.length, self.allowed, self.invariant)

    __rmul__ = __mul__

    def __truediv__(self, c: 'int | float') -> 'Path':
        return Path(self._graph, self.coefficients / c, self.length, self.allowed, self.invariant) # type: ignore

    def __sub__(self, other: 'Path') -> 'Path':
        return self + (other * -1)


    def to_non_allowed(self) -> 'Path':
        if not self.allowed:
            return self
        new_coefficients = self._graph._get_coef_shape(self.length, False)
        paths = self._graph.list_paths(self.length, self.allowed)
        for coef, path in zip(self.coefficients, paths): # type: ignore
            new_coefficients[self._graph._path_index(path, False)] = coef
        return Path(self._graph, new_coefficients, self.length, False)

    def is_allowed(self) -> bool:
        if self.allowed:
            return True
        paths = self._graph.list_paths(self.length, False)
        for i in self._graph._non_allowed_ix(self.length):
            if abs(self.coefficients[self._graph._path_index(paths[i], self.allowed)]) > _params['eps']:
                return False
        return True

    def to_allowed(self, restrict=False) -> 'Path':
        assert restrict or self.is_allowed(), "Has non-zero non-allowed terms and can't be converted."
        ix = [self._graph._path_index(path, self.allowed) for path in self._graph.list_paths(self.length, True)]
        return Path(self._graph, self.coefficients[ix], self.length, True)

    def is_invariant(self) -> bool:
        if self.invariant:
            return True
        return self.d().is_allowed()

    def d(self, regular: bool = True):
        new_coefficients = self._graph.get_d_matrix(self.length, allowed=self.allowed,
                                                             regular=regular,
                                                             invariant=self.invariant) @ self.coefficients
        return Path(self._graph, new_coefficients, self.length - 1, self.invariant, self.invariant)
