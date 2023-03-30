import numpy as np
import time

from encoding import multiprocessing_index,index,beam_search,CoordinateDes
from scannAQ_function import timefn,par_compute_H
from newloss import score_aware_loss

class ScannAQ:
    def __init__(self, data, parameter):
        self.M = parameter.M
        self.K = parameter.K
        self.poolSize = parameter.poolSize
        self.encodeidea = parameter.encodeidea
        self.N = parameter.N
        self.nor = parameter.nor
        self.n = data.shape[0]
        self.D = data.shape[1]
        self.data = data

        self.H = np.zeros((self.n, 2))
        if self.nor:
            self.eta = (self.D - 1) * parameter.T ** 2 / (1 - parameter.T ** 2)
        else:
            self.T = parameter.T*parameter.norm

            self.H = par_compute_H(data,self.T)
    # @profile
    def quantization_error(self, C, Ind, parameter):
        error = 0
        train_data = self.data
        H = self.H
        for i in range(self.n):
            x = train_data[i, ...].reshape(self.D, 1)
            weight = H[i, ...]
            Q_x = np.sum(C[...,Ind[i]],axis=1)
            error = error + score_aware_loss(x, weight, Q_x, parameter)
        return error

    # @profile
    def codebook_train(self, train_data, C, parameter, maxiter=20):
        print("begin codebook_train")
        error_threshold = 0.01
        poolSize = self.poolSize
        print(f"poolSize={poolSize}")  #
        # T = parameter.T
        M = parameter.M
        K = parameter.K
        # N = parameter.N

        D = self.D
        weight_H = self.H
        I = np.eye(D)

        time01 = time.time()  #
        S_ind_Matrix = np.zeros((self.n,M), dtype = int)

        Ind_Matrix,S_ind_Matrix = multiprocessing_index(train_data, weight_H, C, parameter, poolSize)
        
        newerror = self.quantization_error(C, Ind_Matrix, parameter)

        time02 = time.time()  #

        error_change = newerror
        print(f"init error {error_change}")  #
        print(f"first index cost {time02 - time01}s")
        iter_num = 0  
        
        ###
        if self.nor==0:
            normalized_train_data = train_data / np.linalg.norm(train_data, axis=1)[:, np.newaxis]
            h0_set = self.H[:,0:1]-self.H[:,-1:]
        ###

        while error_change > error_threshold * newerror and iter_num < maxiter:
            time1 = time.time()
            iter_num += 1
            olderror = newerror
            num = 0  
            err_change = newerror
            enderr = newerror
            while num < 5 and err_change > 0.01 * enderr:
                num += 1
                initerr = enderr
                for codeword_k in range(M * K):
                    H = np.zeros((D, D))
                    r = np.zeros((D, 1))

                    nu = 0
                    if self.nor:
                        codebook_position = codeword_k // K
                        Ind = np.where(Ind_Matrix[..., codebook_position] == codeword_k)[0] 
                        X_k = train_data[Ind,...]
                        sum_num = X_k.shape[0]
                        H = (self.eta - 1)*X_k.T@X_k + sum_num*I 
                        r1 = self.eta*np.sum(X_k, axis = 0)
                        r1 = r1.reshape(D, 1)
                        r2 = 0
                        for i in Ind:
                            x = train_data[i, ...].reshape(D, 1)
                            ind = Ind_Matrix[i]
                            S = np.sum(C[..., ind], axis=1).reshape(D, 1) - C[..., codeword_k].reshape(D, 1)
                            r2 = r2 - (self.eta - 1) * x @ (x.T @ S) - S
                        r = r1 +r2
                        if sum_num > 0:
                            nu += 1
                    else:
                        codebook_position = codeword_k // K
                        Ind = np.where(Ind_Matrix[..., codebook_position] == codeword_k)[0] 

                        ### 
                        X_k = train_data[Ind,...]
                        h0 = h0_set[Ind,...]
                        normalized_X_k = normalized_train_data[Ind,...]
                        H = normalized_X_k.T@(h0*normalized_X_k)+np.sum(weight_H[Ind,-1:])*I
                        r1 = np.sum(weight_H[Ind,0:1]*X_k,axis=0)
                        r1 = r1.reshape(D, 1)
                        r2 = 0
                        ###


                        for i in Ind:
                            x = train_data[i, ...].reshape(D, 1)

                            ###
                            normalized_x = normalized_train_data[i,...].reshape(D, 1)
                            ###

                            h1, h2 = self.H[i,...]
                            # H = H + (h1 - h2) * x @ x.T / np.vdot(x,x) + h2 * I

                            # S = C@W[i,...].reshape(M*K,1)-C[...,codeword_k].reshape(D,1)
                            ind = Ind_Matrix[i] # ind = np.where(W[i, ...] == 1)[0]
                            S = np.sum(C[..., ind], axis=1).reshape(D, 1) - C[..., codeword_k].reshape(D, 1)

                            # r = r + h1 * x - (h1 - h2) * x @ (x.T @ S) / np.vdot(x,x) - h2 * S
                            
                            ###
                            r2 = r2 - (h1 - h2) * normalized_x @ (normalized_x.T @ S) - h2 * S
                            ### 
                            if h1 > 0:
                                nu += 1
                        ###
                        r = r1 + r2
                        ###
                    ##
                    
                    if nu > 0:
                        c = np.linalg.inv(H) @ r
                        C[..., codeword_k] = c.T
                enderr = self.quantization_error(C, Ind_Matrix, parameter)
                err_change = initerr - enderr  

            time2 = time.time()  #
            Ind_Matrix,S_ind_Matrix = multiprocessing_index(train_data, weight_H, C, parameter, poolSize)

            newerror = self.quantization_error(C, Ind_Matrix, parameter)
            error_change = olderror - newerror
            print(f"iter_num:{iter_num} newerror={newerror},error_change={error_change}({round(error_change/newerror,2)*100}%)")
            time3 = time.time()  #
            print(f"matrix computation cost {time2 - time1}s,index cost {time3 - time2}s")  #
        print("end of the codebook train")
        return C, Ind_Matrix, S_ind_Matrix

    # def test(self, train_data, C, parameter, maxiter=20):
    #     index(0, train_data, self.H, C, parameter, self.poolSize)
