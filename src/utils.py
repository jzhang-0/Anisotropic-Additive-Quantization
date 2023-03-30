import numpy as np
import time
from functools import wraps
from scipy import integrate


def codebook_init(dataset, M, Ks):
    randomI = np.arange(dataset.shape[0])
    np.random.seed(15)
    np.random.shuffle(randomI)
    randomI = np.sort(randomI[0:Ks])
    D = dataset.shape[1]
    Ds = D // M
    Codebook = dataset[randomI, ...]

    C = np.zeros((D * Ks, 1))

    for i in range(M):
        Ci = Codebook[..., i * Ds:(i + 1) * Ds].reshape(Ds * Ks, 1)
        C[i * Ks * Ds:(i + 1) * Ks * Ds] = Ci

    return C


def codebook_init_aq(dataset, M, K):
    randomI = np.arange(dataset.shape[0])
    np.random.seed(15)
    np.random.shuffle(randomI)
    randomI = np.sort(randomI[0:M * K])
    C = dataset[randomI, ...].T
    return C


def residual_set(dataset, inv_tab, code_book):
    """
    return residualset
    """
    residualset = np.zeros(dataset.shape)
    for centroid_id, centroid in zip(inv_tab, code_book.T):
        residualset[centroid_id, :] = dataset[centroid_id, :] - centroid
    return residualset


def f(x, d):
    return np.sin(x) ** d


def h(x, T):
    if T < np.linalg.norm(x, ord=2):
        d = x.size
        alpha = np.arccos(T / np.linalg.norm(x, ord=2))
        Id, _ = integrate.quad(f, 0, alpha, args=d)
        Id2, _ = integrate.quad(f, 0, alpha, args=d - 2)
        h1 = Id2 - Id
        h2 = Id / (d - 1)
        return h1 * 100, h2 * 100
    else:
        return 0, 0


# @profile
def s_ind_transform_ii(s_ind, M, Ks, Ds, Ad):
    a0 = Ad * Ks + s_ind[Ad]
    # a1 = a0 + 1
    a0 = a0 * Ds
    # a1 = a1*Ds
    ii = np.zeros(M * Ds, dtype=int)
    for i in range(M):
        start = a0[i]
        stop = start + Ds
        ii[i * Ds:(i + 1) * Ds] = np.arange(start, stop)
    return ii


def select_testdata(query, dataset, start, end, option):
    if option == 0:
        testdata = query[start:end, ...]
    else:
        testdata = dataset[start:end, ...]
    return testdata

def timefn(fn):
    @wraps(fn)  
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"{fn.__name__} took {t2 - t1} seconds")
        return result

    return measure_time


def start_end_index(n, group_number):
    each_batch_number = n // group_number
    start_index = np.array([i * each_batch_number for i in range(group_number)])
    end_index = start_index + each_batch_number
    end_index[group_number - 1] = n
    return start_index, end_index
