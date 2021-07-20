"""
Sample code automatically generated on 2021-07-16 07:57:35

by geno from www.geno-project.org

from input

parameters
  matrix A
variables
  matrix C
  matrix S
min
  norm2(A-C*S).^2
st
  C >= 0
  S >= 0


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
import numpy as np


try:
    from geno.genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)

class GenoNLP:
    def __init__(self, A, CInit, SInit):
        self.A = A
        self.CInit = CInit
        self.SInit = SInit
        assert isinstance(A, np.ndarray)
        dim = A.shape
        assert len(dim) == 2
        self.A_rows = dim[0]
        self.A_cols = dim[1]
        assert isinstance(CInit, np.ndarray)
        dim = CInit.shape
        assert len(dim) == 2
        self.C_rows = dim[0]
        self.C_cols = dim[1]
        assert isinstance(SInit, np.ndarray)
        dim = SInit.shape
        assert len(dim) == 2
        self.S_rows = dim[0]
        self.S_cols = dim[1]
        self.C_size = self.C_rows * self.C_cols
        self.S_size = self.S_rows * self.S_cols
        # the following dim assertions need to hold for this problem
        assert self.A_rows == self.C_rows
        assert self.A_cols == self.S_cols
        assert self.S_rows == self.C_cols

    def getBounds(self):
        bounds = []
        bounds += [(0, inf)] * self.C_size
        bounds += [(0, inf)] * self.S_size
        return bounds

    def getStartingPoint(self):
        return np.hstack((self.CInit.reshape(-1), self.SInit.reshape(-1)))

    def variables(self, _x):
        C = _x[0 : 0 + self.C_size]
        C = C.reshape(self.C_rows, self.C_cols)
        S = _x[0 + self.C_size : 0 + self.C_size + self.S_size]
        S = S.reshape(self.S_rows, self.S_cols)
        return C, S

    def fAndG(self, _x):
        C, S = self.variables(_x)
        T_0 = (self.A - (C).dot(S))
        f_ = (np.linalg.norm(T_0, 'fro') ** 2)
        g_0 = -(2 * (T_0).dot(S.T))
        g_1 = -(2 * (C.T).dot(T_0))
        g_ = np.hstack((g_0.reshape(-1), g_1.reshape(-1)))
        return f_, g_

def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

def solve(A, CInit, SInit):
    start = timer()
    NLP = GenoNLP(A, CInit, SInit)
    x0 = NLP.getStartingPoint()
    bnds = NLP.getBounds()
    tol = 1E-6
    # These are the standard GENO solver options, they can be omitted.
    options = {'tol' : tol,
               'constraintsTol' : 1E-4,
               'maxiter' : 1000,
               'verbosity' : 1  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.0.3')
        result = minimize(NLP.fAndG, x0,
                          bounds=bnds, options=options)
    else:
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=bnds)

    # assemble solution and map back to original problem
    x = result.x
    C, S = NLP.variables(x)
    solution = {}
    solution['success'] = result.success
    solution['message'] = result.message
    solution['fun'] = result.fun
    solution['grad'] = result.jac
    solution['C'] = C
    solution['S'] = S
    solution['elapsed'] = timer() - start
    return solution

def generateRandomData():
    np.random.seed(0)
    A = np.random.randn(3, 3)
    CInit = np.random.randn(3, 3)
    SInit = np.random.randn(3, 3)
    return A, CInit, SInit

if __name__ == '__main__':
    print('\ngenerating random instance')
    A, CInit, SInit = generateRandomData()
    print('solving ...')
    solution = solve(A, CInit, SInit)
    print('*'*5, 'solution', '*'*5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable C = ', solution['C'])
        print('optimal variable S = ', solution['S'])
        print('solving took %.3f sec' % solution['elapsed'])
