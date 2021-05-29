from typing import Any, Dict, List, NewType, Tuple


Vertex = NewType('Vertex', Any)
ListOfEdges = List[Tuple[Vertex, Vertex]]
Adjacency = Dict[Vertex, List[Vertex]]
EPath = Tuple[Vertex, ...]