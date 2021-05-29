# Path homology toolkit


**<h3 style="color: red"> This is a development branch. None of the code in this branch is expected to work! The code that works, will be occasionally pushed to the main branch. </h3>**

This package provides the algorithm for computing path homology [[1]](#1) and basic path operations. 

## Features

The module contains the ``Graph`` class. It can be initialized from a dict of lists or a numpy matrix. The graph has several methods that compute the space of ∂-invariant paths and the space of n-cycles which are all instances of the ``Path`` class.

The ``Path`` class implements the concept of the path as described in [[1]](#1). It implements both allowed and arbitrary paths and supports basic arithmetic.

Currently, both regular and non-regular homology are supported. By default, the non-reduced homology is computed, however the algorithm can be configured to compute the reduced homology (see Parameters).


## Development

Currently working on the following features:

- ``PathComplex`` class:
    - [v] Create ``BasePathComplex`` class with ``_enum_all_paths()`` and ``_enum_allowed_paths()`` virtual methods.
    - [v] Implement path homology for graphs through PathComplex. Graph contains ``GraphPathComplex`` attribute.
- Finite fields:
    - [v] Write ``null_space()`` for finite fields.
    - [*] Implement ``order`` parameter for path homology related methods.
    - [ ] Benchmark.
- Performance:
    - [ ] Explore xla SVD from jax for faster ``null_space()``.
- Enchancements:
    - [ ] Work on typing.
    - [ ] Review code style and structure.
    - [ ] Reslove imports.
    - [ ] Documentation.
    - [ ] Polish README.md
    - [ ] Sort out params.
- Bright future:
    - [ ] persistent path homology
    - [ ] Path homology over rings + SymPy integration


## Installation

clone the repository and run the following line

    pip install .

## Usage

## Acknowledgements

International Laboratory of Algebraic Topology and its Applications, Higher School of Economics https://cs.hse.ru/en/ata-lab/about

## Parameters

The package provides global options that control the behavior of certain functions. These include the tolerance, the type of homology (i.e. reduced or unreduced) and output options fro the paths.

## References
<a id="1">[1]</a>
Grigor'yan, A., Lin, Y., Muranov, Y., & Yau, S. T. (2012). Homologies of path complexes and digraphs. arXiv preprint arXiv:1207.2834.



