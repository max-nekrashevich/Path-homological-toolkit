# Path homology toolkit

This package provides the algorithm for computing path homology [[1]](#1) and basic path operations. 

## Features

The module contains the ``Graph`` class. It can be initialized from a dict of lists or a numpy matrix. The graph has several methods that compute the space of ∂-invariant paths and the space of n-cycles which are all instances of the ``Path`` class.

The ``Path`` class implements the concept of the path as described in [[1]](#1). It implements both allowed and arbitrary paths and supports basic arithmetic.

Currently, both regular and non-regular homology are supported. By default, the non-reduced homology is computed, however the algorithm can be configured to compute the reduced homology (see Parameters).

## Installation

clone the repository and run the following line

    pip install .

## Acknowledgements

TBD

## Parameters

The package provides global options that control the behavior of certain functions. These include the tolerance, the type of homology (i.e. reduced or unreduced) and output options fro the paths.

## References
<a id="1">[1]</a>
Grigor'yan, A., Lin, Y., Muranov, Y., & Yau, S. T. (2012). Homologies of path complexes and digraphs. arXiv preprint arXiv:1207.2834.