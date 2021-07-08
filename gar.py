# POC: Anish Acharya <anishacharya@utexas.edu>

import numpy as np
import torch
from typing import List
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gar(gar='mean'):
    gar = gar
    print('--------------------------------')
    print('Initializing {} GAR'.format(gar))
    print('--------------------------------')
    if gar == 'mean':
        return Mean()
    else:
        raise NotImplementedError


class GAR:
    """
    This is the base class for all the implemented GAR
    """

    def __init__(self,):
        self.current_losses = []
        self.agg_time = 0
        self.num_iter = 0  # usually if Sub routine has iters ex - GM

    def aggregate(self, G: np.ndarray, ix: List[int] = None, axis=0) -> np.ndarray:
        """
        G: Gradient Matrix where each row is a gradient vector (g_i)
        ix: Columns specified to be aggregated on (if None done on full dimension)
        """
        raise NotImplementedError

    @staticmethod
    def weighted_average(stacked_grad: np.ndarray, alphas=None):
        """
        Implements weighted average of grad vectors stacked along rows of G
        If no weights are supplied then its equivalent to simple average
        """
        n, d = stacked_grad.shape  # n is treated as num grad vectors to aggregate, d is grad dim
        if alphas is None:
            # make alpha uniform
            alphas = [1.0 / n] * n
        else:
            assert len(alphas) == n

        agg_grad = np.zeros_like(stacked_grad[0, :])

        for ix in range(0, n):
            agg_grad += alphas[ix] * stacked_grad[ix, :]
        return agg_grad


class Mean(GAR):

    def __init__(self):
        GAR.__init__(self)

    def aggregate(self, G: np.ndarray, ix: List[int] = None, axis=0) -> np.ndarray:
        # if ix given only aggregate along the indexes ignoring the rest of the ix
        t0 = time.time()
        if ix is not None:
            if axis == 0:
                g_agg = np.zeros_like(G[0, :])
                G = G[:, ix]
                low_rank_mean = self.weighted_average(stacked_grad=G)
                g_agg[ix] = low_rank_mean
            elif axis == 1:
                G = G[ix, :]
                low_rank_mean = self.weighted_average(stacked_grad=G)
                g_agg = low_rank_mean
            else:
                raise ValueError("Wrong Axis")

            self.agg_time = time.time() - t0
            return g_agg
        else:
            t0 = time.time()
            g_agg = self.weighted_average(stacked_grad=G)
            self.agg_time = time.time() - t0
            return g_agg
