from time import time
import numpy as np
import os

from scann_class import ScaNN
from joblib import Parallel, delayed,dump,load
import shutil

from multiprocessing import Pool
from functools import partial
from scipy.linalg import cho_factor, cho_solve
from utils import timefn


class ScaNN_PQ(ScaNN):
    def __init__(self, M, Ks, D, T, nor) -> None:
        super().__init__(D, T, nor)
        self.M = M
        self.Ks = Ks
        self.Ds = self.D// M
        self.suffix = f"PQ_M_{self.M}_K_{self.Ks}" 

    def quantization_error(self, C, data, II, weight_H):
        error = 0
        n = data.shape[0]

        for i in range(n):
            x = data[i, ...].reshape(self.D, 1)
            ii = II[i]
            Q_x = C[ii]
            weight = weight_H[i]
            error = error + self.score_aware_loss(x, Q_x, weight)
        return error


    # @profile
    def single_point_encode(self, C, single_data, weight, iter_num=3) -> tuple:
        # C.shape = (K*D,)
        Ds = self.Ds
        D = self.D
        Ks = self.Ks
        M = self.M

        s_ind = np.zeros(M, dtype=int)
        q_data = np.zeros((D, 1))
        for i in range(M):
            q_data[i * Ds:(i + 1) * Ds, ...] = C[(i * Ks + s_ind[i]) * Ds:(i * Ks + s_ind[i] + 1) * Ds]

        counter = 0
        while counter < iter_num:
            counter += 1
            for subcb in range(M):
                Q_errorlist = np.zeros(Ks)
                for i in range(Ks):
                    s_ind[subcb] = i
                    q_data[subcb * Ds:(subcb + 1) * Ds, ...] = C[(subcb * Ks + i) * Ds:(subcb * Ks + i + 1) * Ds]
                    loss = self.score_aware_loss(single_data, q_data, weight)
                    Q_errorlist[i] = loss

                s_ind[subcb] = np.argmin(Q_errorlist)
                q_data[subcb * Ds:(subcb + 1) * Ds, ...] = C[(subcb * Ks + s_ind[subcb]) * Ds:(subcb * Ks + s_ind[
                    subcb] + 1) * Ds]

        ii = [i * Ds * Ks + s_ind[i] * Ds + j for i in range(M) for j in range(Ds)]

        return s_ind, ii

    # @profile
    # not use
    def _index(self, num_batch, C, data, weight_H):
        """
        data:n*D
        num_batch:Batch of data
        """
        print(time())
        print(os.getpid())
        n, D = data.shape
        poolSize = self.poolSize
        batch = n // poolSize
        assert n % poolSize == 0
        index_data = data[num_batch * batch:(num_batch + 1) * batch, ...]
        weight_H_batch = weight_H[num_batch * batch:(num_batch + 1) * batch, ...]

        S_ind = np.zeros((batch, self.M), dtype=int)
        II = np.zeros((batch, self.D), dtype=int)

        for i in range(batch):
            x = index_data[i, ...].reshape(D, 1)
            weight = weight_H_batch[i,...]
            s_indi, ii_i = self.single_point_encode(C,x,weight)

            S_ind[i] = s_indi
            II[i] = ii_i
        return S_ind, II

    # not use
    @timefn
    def multiprocessing_index(self, C, data, weight_H, poolSize):
        self.poolSize = poolSize

        n = data.shape[0]
        batch = n // self.poolSize
        S_ind = np.zeros((n, self.M), dtype=int)
        II = np.zeros((n, self.D), dtype=int)
        
        ff = partial(self._index, C = C, data=data, weight_H= weight_H)

        p = Pool(self.poolSize)
        a = p.map_async(ff, range(self.poolSize)).get()
        p.close()
        p.join()

        for i in range(self.poolSize):
            S_ind[i * batch:(i + 1) * batch, ] = a[i][0]
            II[i * batch:(i + 1) * batch, ] = a[i][1]

        return S_ind, II
    
    @timefn
    def index(self, C, data, weight_H, njobs = os.cpu_count()//2):
        n, D = data.shape
        
        S_ind = np.zeros((n, self.M), dtype=int)
        II = np.zeros((n, self.D), dtype=int)
        
        ### 
        # for i in range(n):
        #     x = data[i, ...].reshape(D, 1)
        #     weight = weight_H[i,...]
        #     s_indi, ii_i = self.single_point_encode(C,x,weight)

        #     S_ind[i] = s_indi
        #     II[i] = ii_i
        ###
        
        ### 
        result = Parallel(n_jobs = njobs,backend='multiprocessing')(delayed(self.single_point_encode)(C,x,weight) for x,weight in zip(data,weight_H))
        for i in range(n):
            S_ind[i],II[i] = result[i]
        ### 

        return S_ind, II

    def _write_code_memmap(self, dataset_memmap, H_memmap, idx, C, S_ind_memmap, II_memmap):
        S_ind_memmap[idx],II_memmap[idx] = self.single_point_encode(C,dataset_memmap,H_memmap)


    @timefn
    # @profile
    def index_opt(self, C, data, weight_H, njobs = 20):
        folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        data_filename_memmap = os.path.join(folder, 'data_memmap')
        dump(data, data_filename_memmap)
        dataset_memmap = load(data_filename_memmap, mmap_mode='r')

        weight_H_filename_memmap = os.path.join(folder, 'H_memmap')
        dump(weight_H, weight_H_filename_memmap)
        H_memmap = load(weight_H_filename_memmap, mmap_mode='r')

        output_S_ind_filename_memmap = os.path.join(folder, 'output_S_ind_memmap')
        output_II_filename_memmap = os.path.join(folder, 'output_II_memmap')

        n = dataset_memmap.shape[0]
        II_memmap = np.memmap(output_II_filename_memmap, dtype=int, shape=(n, self.D), mode='w+')    
        S_ind_memmap = np.memmap(output_S_ind_filename_memmap, dtype=int, shape=(n, self.M), mode='w+')    

        Parallel(n_jobs = njobs)(delayed(self._write_code_memmap)(dataset_memmap[idx], H_memmap[idx], idx, C, S_ind_memmap, II_memmap) for idx in range(n))
        S_ind = np.array(S_ind_memmap)
        II = np.array(II_memmap)

        try:
            shutil.rmtree(folder)
        except:  
            print('Could not clean-up automatically.')

        return S_ind, II


    @timefn
    def compute_TR(self,data,II,weigh_H):
        Ks = self.Ks
        D = self.D
        n = data.shape[0]
        if self.nor:
            Ts = np.zeros((Ks * D, Ks * D), dtype="float32")
            Rs = np.zeros((Ks * D, 1), dtype="float32")
            for i in range(n):
                x = data[i].reshape(self.D,1)
                ii = II[i]            
                Ts[ii,ii] += 1
                Ts[np.ix_(ii,ii)] += (self.eta-1)*x@x.T
                Rs[ii] += self.eta*x
        else:
            Ts = np.zeros((Ks * D, Ks * D), dtype="float32")
            Rs = np.zeros((Ks * D, 1), dtype="float32")
            for i in range(n):
                h1,h2 = weigh_H[i]
                x = data[i].reshape(self.D,1)
                s = np.vdot(x,x)
                ii = II[i]            
                Ts[ii,ii] += h2
                Ts[np.ix_(ii,ii)] += (h1-h2)*x@x.T/s
                Rs[ii] += h1*x
        return Ts,Rs

    def codebook_updata(self,train_data,II,weigh_H):
        Ts,Rs = self.compute_TR(train_data,II,weigh_H)
        c0, low = cho_factor(Ts)  # O(K**3 D**3) 
        C = cho_solve((c0, low), Rs)
        return C
    
    def save_TR(self,train_data,II,weight_H):
        Ts,Rs = self.compute_TR(train_data,II,weight_H)
        np.save(f"./save/Ts_{train_data.shape[0] // 1000}k_{self.suffix}",Ts)
        np.save(f"./save/Rs_{train_data.shape[0] // 1000}k_{self.suffix}",Rs)


    def train(self, train_data, C, saveTR=1, maxiter=20):
        """
        return C, S_ind, II
        """
        data = train_data
        weight_H,_ = self.compute_H_Direction(train_data)

        print("begin codebook train")
        error_threshold = 0.01

        # S_ind, II = self.index_opt(C, data, weight_H)
        S_ind, II = self.index(C, data, weight_H)

        newerror = self.quantization_error(C, data, II, weight_H)

        error_change = newerror
        print(f"init quantization loss = {error_change}")

        iter_num = 0
        while error_change > error_threshold * newerror and iter_num < maxiter:
            iter_num += 1
            print(f"iteration number {iter_num}")
            olderror = newerror

            C = self.codebook_updata(train_data,II,weight_H)
            S_ind, II = self.index(C, data, weight_H)
            # S_ind, II = self.index_opt(C, data, weight_H)

            newerror = self.quantization_error(C, data, II, weight_H)
            error_change = olderror - newerror
            print(f"The new loss {newerror},The loss is reduced {error_change}({round(error_change/newerror*100,2)}%)")  #

        if saveTR==1:
            self.save_TR(train_data,II,weight_H)
        print("end of the codebook train")
        return C, S_ind, II
 