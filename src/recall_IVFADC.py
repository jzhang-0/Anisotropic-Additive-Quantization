import time
from utils import timefn
import numpy as np

from recall_fix import Recall_PQ, Recall_AQ, Recall


class Recall_IVFADC(Recall):
    def __init__(self, inv_tab, code_book, queries) -> None:
        self.inv_tab = inv_tab
        self.code_book = code_book
        self.queries = queries

        l = len(inv_tab)
        self.count = np.zeros(l)
        for i in range(l):
            self.count[i] = len(inv_tab[i])

        inner_M = queries @ code_book
        c_id = np.argsort(inner_M, axis=1)
        self.c_id = np.fliplr(c_id)

        row = np.arange(queries.shape[0])[:, np.newaxis]
        self.inner_1 = inner_M[row, self.c_id]  

    def transform(self, c_i):
        """
        c_i: 类中心的索引|单个query要找的类
        return: q_ci
        """
        q_ci = []
        for i in c_i:
            q_ci += self.inv_tab[i]
        return q_ci

    # 2000个类搜其中100个
    def search_cluster_centroids(self, code_book, queries, num_centroids_to_search):
        """
        code_book:每个中心为列向量（32*500）
        """
        inner_M = queries @ code_book
        c_id = np.argsort(inner_M, axis=1)[:, -num_centroids_to_search:]
        c_id = np.fliplr(c_id)

        # 从inner_M中读取最大的内积
        row = np.arange(queries.shape[0])[:, np.newaxis]
        inner_1 = inner_M[row, c_id]

        return c_id, inner_1  # c_id:the index of argsort inner(queries,code_book)

    def searched_centroid_num(self, q_c_id, num_to_search):
        len = 0

        c_num = 0
        while len < num_to_search:
            id = q_c_id[c_num]
            # centroid_index += self.inv_tab[id]
            len += self.count[id]
            c_num += 1
            assert c_num <= q_c_id.size

        return c_num

    def query_mips(self,q,dataset):
        inner = -dataset@q
        return inner.argsort()   

    @timefn
    def rerank(self, dataset, queries, neighbors_matrix, topk=10):
        nq = queries.shape[0]
        result = np.zeros((nq,topk),dtype=np.int)
        for i,q,neighbors in zip(range(nq),queries,neighbors_matrix):
            index = self.query_mips(q, dataset[neighbors]) 
            result[i] = neighbors[index[0:topk]]
        
        return result


class Recall_IVFADC_PQ(Recall_PQ, Recall_IVFADC):
    def __init__(self, M, Ks, D, inv_tab, code_book, queries) -> None:
        Recall_PQ.__init__(self, M, Ks, D)
        Recall_IVFADC.__init__(self, inv_tab, code_book, queries)

    def search_neightbor_IVFADC(self, C, S_ind, query, c_i, q_inner_1, top_k=64):
        '''
        c_i = c_id[query_id]  
        return: adc inner, np.array
        '''
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

        n = S_ind.shape[0]
        inner = np.zeros(n)
        for centroid_id, inner_1 in zip(c_i, q_inner_1):
            index = self.inv_tab[centroid_id]
            quantization_inner_2 = np.sum(Table[i1, S_ind[index, :]], axis=1)  
            inner[index] = inner_1 + quantization_inner_2

        q_ci = self.transform(c_i)
        assert len(q_ci) > top_k
        bool = np.full(n, True)
        bool[q_ci] = False
        inner[bool] = -np.inf

        ind = np.argpartition(inner, -top_k)[-top_k:]
        return ind[np.argsort(inner[ind])]

    # @profile
    def search_neightbor_IVFADC_fixnum_opt(self, C, S_ind, query, c_i, q_inner_1, num_to_search, inner, top_k=64, timeinfo=0):
        M = self.M
        Ks = self.Ks
        Ds = self.Ds

        # t1 = time.time()
        q = query.reshape(M, Ds)
        Table = np.matmul(C, q[:, :, np.newaxis])[:, :, 0]
        # t2 = time.time()

        # if timeinfo:
        #     print(f"build table cost time {t2-t1} s")

        # t3 = time.time()
        i1 = np.arange(M)

        # n = S_ind.shape[0]
        # inner = np.zeros(n)

        num = self.searched_centroid_num(c_i, num_to_search=num_to_search)
        c_i = c_i[0:num]
        q_inner_1 = q_inner_1[0:num]

        for centroid_id, inner_1 in zip(c_i, q_inner_1):
            index = self.inv_tab[centroid_id]
            quantization_inner_2 = np.sum(Table[i1, S_ind[index, :]], axis=1) 
            inner[index] = inner_1 + quantization_inner_2

        q_ci = self.transform(c_i)
        assert len(q_ci) >= top_k

        Ir = inner[q_ci]
        ind_ = np.argpartition(Ir, -top_k)[-top_k:]
        ind__ = ind_[np.argsort(Ir[ind_])]

        q_ci_npy = np.array(q_ci)
        ind = q_ci_npy[ind__]
        # t4 = time.time()
        # if timeinfo:
        #     print(f"other cost time {t4-t3} s")
        return ind

    # @profile
    def search_neightbor_IVFADC_fixnum(self, C, S_ind, query, c_i, q_inner_1, num_to_search, top_k=64):
        '''
        return: adc inner, np.array ndim=1
        '''
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

        n = S_ind.shape[0]
        inner = np.zeros(n)

        num = self.searched_centroid_num(c_i, num_to_search=num_to_search)
        c_i = c_i[0:num]
        q_inner_1 = q_inner_1[0:num]

        for centroid_id, inner_1 in zip(c_i, q_inner_1):
            index = self.inv_tab[centroid_id]
            quantization_inner_2 = np.sum(Table[i1, S_ind[index, :]], axis=1)  
            inner[index] = inner_1 + quantization_inner_2

        q_ci = self.transform(c_i)
        assert len(q_ci) >= top_k

        Ir = inner[q_ci]
        ind_ = np.argpartition(Ir, -top_k)[-top_k:]
        ind__ = ind_[np.argsort(Ir[ind_])]

        q_ci_npy = np.array(q_ci)
        ind = q_ci_npy[ind__]
        return ind

    @timefn
    def search_neightbors_IVFADC(self, C, S_ind, queries, c_id, inner_1, topk=64):
        n = queries.shape[0]
        Neighbor_Matrix = np.zeros((n, topk), dtype=int)

        for q, c_i, q_inner_1, i in zip(queries, c_id, inner_1, range(n)):
            Neighbor_Matrix[i] = self.search_neightbor_IVFADC(C, S_ind, q, c_i, q_inner_1, top_k=topk)

        return Neighbor_Matrix

    @timefn
    def search_neightbors_IVFADC_fixnum(self, C, S_ind, queries, num_to_search, topk=64):
        print(f"number to search:{num_to_search}")
        n = queries.shape[0]
        Neighbor_Matrix = np.zeros((n, topk), dtype=int)

        for q, c_i, q_inner_1, i in zip(queries, self.c_id, self.inner_1, range(n)):
            Neighbor_Matrix[i] = self.search_neightbor_IVFADC_fixnum(C, S_ind, q, c_i, q_inner_1, num_to_search,
                                                                     top_k=topk)

        return Neighbor_Matrix

    @timefn
    # @profile
    def search_neightbors_IVFADC_fixnum_opt(self, C, S_ind, queries, num_to_search, topk=64):
        if topk > num_to_search:
            topk = num_to_search
        print(f"number to search:{num_to_search}")
        n = queries.shape[0]
        Neighbor_Matrix = np.zeros((n, topk), dtype=int)
        C = C.reshape(self.M, self.Ks, self.Ds)

        inner = np.zeros(S_ind.shape[0])
        for q, c_i, q_inner_1, i in zip(queries, self.c_id, self.inner_1, range(n)):
            Neighbor_Matrix[i] = self.search_neightbor_IVFADC_fixnum_opt(C, S_ind, q, c_i, q_inner_1, num_to_search, inner,
                                                                         top_k=topk,timeinfo=0)

        return Neighbor_Matrix


class Recall_IVFADC_AQ(Recall_IVFADC, Recall_AQ):
    def __init__(self, inv_tab, code_book, queries) -> None:
        super().__init__(inv_tab, code_book, queries)

    # @profile
    def search_neightbor_IVFADC(self, C, Ind_Matrix, query, c_i, q_inner_1, topk=64):
        n = Ind_Matrix.shape[0]
        Table = C.T @ query
        Table = Table.flatten()
        inner = np.zeros(n)
        for centroid_id, inner_1 in zip(c_i, q_inner_1):  # inner_1 is scale
            index = self.inv_tab[centroid_id]
            quantization_inner_2 = np.sum(Table[Ind_Matrix[index]], axis=1)  
            inner[index] = inner_1 + quantization_inner_2

        q_ci = self.transform(c_i)
        assert len(q_ci) > topk
        bool = np.full(n, True)
        bool[q_ci] = False
        inner[bool] = -np.inf

        ind = np.argpartition(inner, -topk)[-topk:]
        return ind[np.argsort(inner[ind])]

    # @profile
    def search_neightbor_IVFADC_fixnum(self, C, Ind_Matrix, query, c_i, q_inner_1, num_to_search, inner, topk=64, timeinfo=0):
        # time1 = time.time()
        Table = C.T @ query
        # time2 = time.time()

        # if timeinfo:
        #     print(f"build table cost time {time2-time1} s")
        
        # time3 = time.time()
        num = self.searched_centroid_num(c_i, num_to_search=num_to_search)
        c_i = c_i[0:num]
        q_inner_1 = q_inner_1[0:num]

        for centroid_id, inner_1 in zip(c_i, q_inner_1):  # inner_1 is a number
            index = self.inv_tab[centroid_id]
            quantization_inner_2 = np.sum(Table[Ind_Matrix[index]], axis=1)  # array length is the number of i-th category
            inner[index] = inner_1 + quantization_inner_2

        q_ci = self.transform(c_i)
        assert len(q_ci) >= topk

        Ir = inner[q_ci]
        ind_ = np.argpartition(Ir, -topk)[-topk:]
        ind__ = ind_[np.argsort(Ir[ind_])]

        q_ci_npy = np.array(q_ci)
        ind = q_ci_npy[ind__]
        # time4 = time.time()

        # if timeinfo:
        #     print(f"other cost time {time4 - time3} s")
        return ind

    @timefn
    def search_neightbors_IVFADC(self, C, Ind_Matrix, queries, c_id, inner_1, topk=64):
        n, D = queries.shape
        neighbors = np.zeros((n, topk), dtype=int)
        for c_i, q_inner_1, i in zip(c_id, inner_1, range(n)):
            query = queries[i, ...].reshape(D, 1)
            Q_index = self.search_neightbor_IVFADC(C, Ind_Matrix, query, c_i, q_inner_1, topk=topk)
            neighbors[i] = Q_index
        return neighbors

    @timefn
    # @profile
    def search_neightbors_IVFADC_fixnum(self, C, Ind_Matrix, queries, num_to_search, topk=64):
        if topk > num_to_search:
            topk = num_to_search
        print(f"the number of search items(number to search):{num_to_search}")
        n = queries.shape[0]
        neighbors = np.zeros((n, topk), dtype=int)
        inner = np.zeros(Ind_Matrix.shape[0])

        for c_i, q_inner_1, i in zip(self.c_id, self.inner_1, range(n)):
            query = queries[i, ...]
            Q_index = self.search_neightbor_IVFADC_fixnum(C, Ind_Matrix, query, c_i, q_inner_1, num_to_search,inner,
                                                          topk=topk,timeinfo=0)
            neighbors[i] = Q_index
        return neighbors
