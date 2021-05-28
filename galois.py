import galois as gal
import numpy as np

GF = g.GF(2)

def null_space_galois(A):
    M, N = A.shape
    ext_A = np.vstack([A, GF.Identity(N)]).T
    return ext_A.row_reduce(M)[M:, M:].T
