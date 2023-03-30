import random
import numpy as np
from scann_class import ScaNN
from utils import timefn


class ScaNN_VQ(ScaNN):
    def __init__(self, D, T, nor, K_v) -> None:
        super().__init__(D, T, nor)
        self.K_v = K_v

    @timefn
    def _kmeans_clustering_ScaNN(self, dataset, weight_H_set, code_book):
        """
        code_book:(K,D)        
        """
        K = self.K_v
        n = dataset.shape[0]

        dist = self.score_aware_loss_many_to_many(dataset, code_book, weight_H_set)
        index = np.argmin(dist, axis=1)  # size = dataset.shape[0], denoted the centroid id
        inversted_table = [[] for i in range(K)]

        data_index = 0
        for i in index:
            inversted_table[i].append(data_index)  
            data_index += 1

        ii = np.arange(n)
        loss = np.sum(dist[ii, index])

        return inversted_table, loss

    @timefn
    def _kmeans_code_book_updata_ScaNN(self, dataset, weight_H_set, inversted_table):
        K = self.K_v
        D = self.D
        I = np.eye(D)
        code_book = np.zeros((D, K))
        if self.nor:
            for i in range(K):
                index = inversted_table[i]
                num = len(index)
                if num > 0:
                    X = dataset[index, ...]
                    code_word_i = np.linalg.inv(num * I + (self.eta - 1) * X.T @ X) @ (self.eta * np.sum(X, axis=0))
                    code_book[..., i] = code_word_i
        else:
            h0_set = weight_H_set[:, 0:1] - weight_H_set[:, -1:]
            # norm_s_set = np.sum(dataset ** 2, axis=1)[:,np.newaxis]
            normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
            for i in range(K):
                index = inversted_table[i]
                weight_H = weight_H_set[index, ...]
                sum_h = np.sum(weight_H[:, 1])
                if sum_h > 0:
                    X = dataset[index, ...]
                    h0 = h0_set[index, ...]
                    normalized_X = normalized_dataset[index, ...]
                    # norm_s = norm_s_set[index,...]
                    code_word_i = np.linalg.inv(sum_h * I + normalized_X.T @ (h0 * normalized_X)) @ np.sum(
                        weight_H[:, 0:1] * X, axis=0)
                    code_book[..., i] = code_word_i
        return code_book

    def kmeans_ScaNN(self, dataset, weight_H_set, thresh=0.01, max_iter=20):
        """
        code_book : shape=(D,K)
        return inversted_table, code_book, loss
        """
        n = dataset.shape[0]
        K = self.K_v
        random.seed(15)
        ii = random.sample(range(1, n), K)
        # code_book = dataset[ii,...].reshape(self.D, K)
        code_book = dataset[ii, ...].T
        inversted_table, loss = self._kmeans_clustering_ScaNN(dataset, weight_H_set, code_book)
        print(f"init_loss = {loss}")

        loss_change = loss
        s = 0
        while loss_change > loss * thresh and s < max_iter:
            s += 1
            old_loss = loss
            code_book = self._kmeans_code_book_updata_ScaNN(dataset, weight_H_set, inversted_table)
            inversted_table, loss = self._kmeans_clustering_ScaNN(dataset, weight_H_set, code_book)
            loss_change = old_loss - loss
            print(f"iter_num:{s},loss_change={loss_change},new loss = {loss}({round(loss_change / loss, 2) * 100}%)")

        print("end kmeans_ScaNN")

        return inversted_table, code_book, loss
