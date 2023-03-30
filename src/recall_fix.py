import os
from utils import timefn
import numpy as np
from joblib import Parallel, delayed


class Recall:
    def __init__(self) -> None:
        pass

    def compute_true_neighbors(self, target_set, test_set):
        distance = target_set @ test_set
        return np.argmax(distance, axis=0)

    def compute_recall(self, neighbors, true_neighbors):
        total = 0
        for gt_row, row in zip(true_neighbors, neighbors):
            total += np.intersect1d(gt_row, row).shape[0]
        return total / true_neighbors.size


class Recall_PQ(Recall):
    def __init__(self, M, Ks, D) -> None:
        self.M = M
        self.Ks = Ks
        self.D = D
        self.Ds = D // M

    def search_neightbor(self, C, S_ind, query, top_k=64):
        M = self.M
        Ks = self.Ks
        Ds = self.Ds
        Table = np.zeros((M, Ks))

        for i in range(M):
            query_sub = query[i * Ds:(i + 1) * Ds]
            C_i = C[i * Ks * Ds:(i + 1) * Ks * Ds]
            codebook_i = C_i.reshape(Ds, Ks, order="F")
            Table[i] = query_sub.T @ codebook_i

        i1 = np.arange(M)
        quantization_inner = np.sum(Table[i1, S_ind[..., i1]], axis=1)  

        ind = np.argpartition(quantization_inner, -top_k)[-top_k:]
        return ind[np.argsort(quantization_inner[ind])]

    @timefn
    def search_neighbors(self, C, queries, S_ind, topk=512):
        n = queries.shape[0]
        Neighbor_Matrix = np.zeros((n, topk), dtype=int)

        result = Parallel(n_jobs=50,backend="multiprocessing")(delayed(self.search_neightbor)(C, S_ind, q, top_k=topk) for q in queries)
        for i in range(n):
            Neighbor_Matrix[i] = result[i]
        return Neighbor_Matrix

    def compute_recall_PQ(self, testdata, C, S_ind, true_neighbors):
        print(f"the number of queries is {testdata.shape[0]},the number of target set is {S_ind.shape[0]}")
        rk = (1, 10, 20, 32, 64)
        Neighbor_Matrix = self.search_neighbors(C, testdata, S_ind)
        for i in rk:
            Neighbor_M = Neighbor_Matrix[..., -i:]
            result = self.compute_recall(Neighbor_M, true_neighbors)
            print(f"recall@{i} = {result}")


class Recall_AQ(Recall):
    def __init__(self) -> None:
        super().__init__()

    def compute_neighbors(self, C, Ind_Matrix, queries, topk=64):
        n, D = queries.shape
        neighbors = np.zeros((n, topk), dtype=int)
        for i in range(n):
            query = queries[i, ...].reshape(D, 1)
            Q_index = self.Q_inner_A_index(C, Ind_Matrix, query, topk=topk)
            neighbors[i] = Q_index
        return neighbors

    # @profile
    def Q_inner_A_index(self, C, Ind_Matrix, query, topk=64):
        n = Ind_Matrix.shape[0]
        quantization_inner = np.zeros(n)
        Table = C.T @ query
        Table = Table.flatten()

        quantization_inner = np.sum(Table[Ind_Matrix], axis=1)

        ind = np.argpartition(quantization_inner, -topk)[-topk:]
        return ind[np.argsort(quantization_inner[ind])]
