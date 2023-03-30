import numpy as np
import os

from scann_class import ScaNN
from scann_PQ import ScaNN_PQ

class ScaNN_IVFADC(ScaNN):

    def __init__(self, D, T, nor) -> None:
        super().__init__(D, T, nor)

    def residual_set(self,dataset,inv_tab,code_book):
        """
        return residualset
        """
        residualset = np.zeros((dataset.shape))
        for centroid_id, centroid in zip(inv_tab,code_book.T):
            residualset[centroid_id,:] = dataset[centroid_id,:] - centroid
        return residualset


class ScaNN_PQ_IVFADC(ScaNN_PQ,ScaNN_IVFADC):
    def __init__(self, M, Ks, D, T, nor) -> None:
        super().__init__(M, Ks, D, T, nor)