import time
import os
import numpy as np
import shutil

from joblib import dump,load,delayed,Parallel
from newloss import h
from functools import wraps


def codebook_init(dataset,M,K):
    randomI = np.arange(dataset.shape[0])
    np.random.seed(15)
    np.random.shuffle(randomI)
    randomI = np.sort(randomI[0:M * K])
    C = dataset[randomI, ...].T
    return C


def initcompute(C, W, train_data, parameter):
    """
    return:SumH,SumR1,SumR2
    #H(R1-R2@Vec(C))
    """
    M = parameter.M
    K = parameter.K
    T = parameter.T
    D = C.shape[0]
    I = np.eye(D)

    SumH = []
    SumR1 = []
    SumR2 = []
    for codeword_k in range(M * K):
        H = 0
        r1 = 0
        r2 = 0
        for i in np.where(W[..., codeword_k] == 1)[0]:
            x = train_data[i, ...].reshape(D, 1)
            h1, h2 = h(x, T)
            Hi = (h1 - h2) * x @ x.T / np.vdot(x,x) + h2 * I
            H = H + Hi
            r1 = r1 + h1 * x

            w = W[i, ...].copy()
            w[codeword_k] = 0
            r2 = r2 + Hi @ np.kron(w, I)

        SumH.append(H)
        SumR1.append(r1)
        SumR2.append(r2)
    return SumH, SumR1, SumR2

def timefn(fn):
    @wraps(fn)  
    def measure_time(*args,**kwargs):
        t1 = time.time()
        result = fn(*args,**kwargs)
        t2 = time.time()
        print(f"{fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time

def compute_H(dataset,T):
    n = dataset.shape[0]
    H = np.zeros((n, 2))
    for i in range(n):
        x = dataset[i, ...]
        H[i, ...] = h(x, T)
    return H

def _write_H_memmap(dataset_memmap,idx,H_memmap,T):
    H_memmap[idx] = h(dataset_memmap[idx],T)


def par_compute_H(dataset,T):
    folder = './joblib_memmap'
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    data_filename_memmap = os.path.join(folder, 'data_memmap')
    dump(dataset, data_filename_memmap)
    dataset_memmap = load(data_filename_memmap, mmap_mode='r')
    output_filename_memmap = os.path.join(folder, 'output_memmap')

    tic = time.time()
    n = dataset_memmap.shape[0]
    H_memmap = np.memmap(output_filename_memmap, dtype=dataset.dtype, shape=(n, 2), mode='w+')    # dtype=dataset.dtype is very important,otherwise H values will be 0
    Parallel(n_jobs = 20)(delayed(_write_H_memmap)(dataset_memmap,idx,H_memmap,T) for idx in range(n))
    H = np.array(H_memmap)

    toc = time.time()
    print(f"par_compute_H cost {toc - tic} s")

    try:
        shutil.rmtree(folder)
    except:  
        print('Could not clean-up automatically.')

    return H


def pqcodebook_as_init(codebook_pq,M,K,D):
    codebook_pq = codebook_pq.flatten()
    Ds = D // M
    C = np.zeros((D,M*K))
    for i in range(M):
        for k in range(K):
            C_i = np.zeros(D)
            C_i[i*Ds:i*Ds+Ds] = codebook_pq[i*K*Ds+k*Ds:i*K*Ds+k*Ds+Ds]
            C[:,i*K+k]=C_i
    return C

