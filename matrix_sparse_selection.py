# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
Here we explore different co-ordinate sampling ideas.
Essentially, the idea is to instead of taking all the coordinates sub-sample
pre-defined k subset of co-ordinates and aggregate only along these directions.

--- This can be thought of a column sampling of matrix G where each row
corresponds to g_i i.e. gradient vector computed on batch / client i
"""

import numpy as np


class SparseApproxMatrix:
    def __init__(self, sampling_rule=None, axis='dim', beta=1, ef=False):
        self.sampling_rule = sampling_rule  # sampling algo
        axis = axis  # 0: column sampling / dimension , 1: row sampling / clients
        if axis == 'dim':
            self.axis = 0
        elif axis == 'n':
            self.axis = 1
        else:
            raise ValueError
        self.beta = beta  # fraction of ix to sample
        self.k = None  # Number of ix ~ to be auto populated
        self.ef = ef
        print('Error Feedback is: {}'.format(self.ef))
        self.residual_error = 0
        self.normalized_residual = 0

    def sparse_approx(self, G: np.ndarray, lr=1) -> [np.ndarray, np.ndarray]:
        if self.sampling_rule not in ['active_norm', 'random']:
            raise NotImplementedError

        n, d = G.shape
        G_sparse = np.zeros_like(G)

        # for the first run compute k and residual error
        if self.k is None:
            if self.beta < 0:
                raise ValueError
            elif self.axis == 0:
                self.k = int(self.beta * d) if self.beta > 0 else 1
                print('Sampling {} coordinates out of {}'.format(self.k, d))
            elif self.axis == 1:
                self.k = int(self.beta * n) if self.beta > 0 else 1
                print('Sampling {} samples out of {}'.format(self.k, n))

            self.residual_error = np.zeros((n, d), dtype=G[0, :].dtype)

        # Error Compensation (if ef is False, residual error = 0 as its not updated
        G = (lr * G) + self.residual_error

        # Invoke Sampling algorithm
        if self.sampling_rule == 'active_norm':
            I_k = self._active_norm_sampling(G=G)
        elif self.sampling_rule == 'random':
            I_k = self._random_sampling(d=d if self.axis == 0 else n)
        else:
            raise NotImplementedError

        if self.axis == 0:
            G_sparse[:, I_k] = G[:, I_k]
        elif self.axis == 1:
            G_sparse[I_k, :] = G[I_k, :]
        else:
            raise ValueError

        if self.ef is True:
            # update residual error
            delta = G - G_sparse
            memory = np.mean(delta, axis=0)
            self.residual_error = np.tile(memory, (G.shape[0], 1))
            # self.residual_error = G - G_sparse

        return G_sparse / lr, I_k

    # Implementation of different "Matrix Sparse Approximation" strategies
    def _random_sampling(self, d) -> np.ndarray:
        """
        Implements Random (Gauss Siedel) subset Selection
        """
        all_ix = np.arange(d)
        I_k = np.random.choice(a=all_ix,
                               size=self.k,
                               replace=False)

        return I_k

    def _active_norm_sampling(self, G: np.ndarray) -> np.ndarray:
        """
        Implements Gaussian Southwell Subset Selection / Active norm sampling
        Ref: Drineas, P., Kannan, R., and Mahoney, M. W.  Fast monte carlo algorithms for matrices:
        Approximating matrix multiplication. SIAM Journal on Computing, 36(1):132–157, 2006
        """
        # Exact Implementation ~ O(d log d)
        # norm_dist = G.sum(axis=self.axis)
        # norm_dist = np.square(norm_dist)i
        norm_dist = np.linalg.norm(G, axis=self.axis)
        norm_dist /= norm_dist.sum()
        sorted_ix = np.argsort(norm_dist)[::-1]

        I_k = sorted_ix[:self.k]

        mass_explained = np.sum(norm_dist[I_k])
        self.normalized_residual = mass_explained

        return I_k
