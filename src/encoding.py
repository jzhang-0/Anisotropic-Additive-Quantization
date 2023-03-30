import time

import numpy as np
from newloss import score_aware_loss,score_aware_loss_one_to_many
from multiprocessing import Pool
from functools import partial

# @profile
def CoordinateDes(data, weight, C, parameter):
    """
    args:
        data:one data
    return
        aq code
    """
    x = data
    M = parameter.M
    K = parameter.K
    iter_num = parameter.iter_num
    Ad = np.arange(M)
    S_ind = np.zeros(M, dtype=int)
    iter = 0
    while iter < iter_num:
        iter += 1
        ind = S_ind + Ad*K
        S_codeword = np.sum(C[:,ind],axis=1)
        for codebook in range(M):
            Pcodeword = S_codeword - C[:, S_ind[codebook] + codebook * K]

            ###
            Q_errorlist = np.zeros(K)
            for i in range(K):
                curr_Qx = Pcodeword + C[:, i + codebook * K]
                Q_errorlist[i] = score_aware_loss(x, weight, curr_Qx, parameter)
            ###

            ###
            # curr_Qx = Pcodeword.reshape(D,1) + C[:, codebook * K:(codebook+1) * K]
            # Q_errorlist = score_aware_loss_one_to_many(x, weight, curr_Qx, parameter)
            ###
            
            S_ind[codebook] = np.argmin(Q_errorlist)

            S_codeword = Pcodeword + C[:, S_ind[codebook] + codebook * K]

    ind = S_ind + Ad*K  
    return ind,S_ind

def beam_search(data, weight, C, parameter): 
    """
    Quantitative(data)=Cw

    Args:
        data:A D dimensional vector
        C:Current codebook,(D,M*K)
        parameter.M:the numbers of codebook
        parameter.N:Take the topN
        weight: (2,)
    Returns:
        aq code
    """
    D = parameter.D
    K = parameter.K
    M = parameter.M
    N = parameter.N
    Ad = np.arange(M)
    
    ###
    loss = np.array([])
    for t in range(K):
        codeword = C[..., t]
        singleloss = score_aware_loss(data, weight, codeword, parameter)
        loss = np.append(loss, singleloss)
    ###
    # loss = score_aware_loss_one_to_many(data, weight, C[...,0:K], parameter)


    bestpath = np.argsort(loss)[0:N].reshape(N, 1)

    for m in range(1, M):
        error = np.array([])
        for t in range(N):
            current_index = bestpath[t, ...]
            s = 0

            ###
            # selectcode = np.zeros((D, 1))
            # for t1 in current_index:
            #     selectcode = C[..., s * K + t1].reshape(D, 1) + selectcode
            #     s = s + 1
            ###
            length = current_index.size
            s += length
            selectcode = np.sum(C[...,current_index + K*np.arange(length)],axis = 1)
            selectcode = selectcode.reshape(D,1)

            ###
            loss = np.array([])
            for t2 in range(K):
                codeword = C[..., s * K + t2].reshape(D, 1)
                # singleloss = score_aware_loss(data - selectcode, codeword, parameter)
                singleloss = score_aware_loss(data, weight, selectcode + codeword, parameter)
                loss = np.append(loss, singleloss)  
            ###

            # loss = score_aware_loss_one_to_many(data,weight,selectcode + C[...,s*K:(s+1)*K], parameter)

            error = np.append(error, loss)  
        # min_error=np.sort(error)[0]
        minN_error_index = np.argsort(error)[0:N] 

        newbestpath = np.zeros((N, m + 1))
        s = 0
        for t in minN_error_index:
            path = t // K
            selectcode_index = t - t // K * K  
            newbestpath[s] = np.append(bestpath[path], selectcode_index)
            s = s + 1
        bestpath = newbestpath
        bestpath = np.array(bestpath, dtype=int)

    S_ind = bestpath[0]
    ind = S_ind + Ad*K

 
    return ind,S_ind

# @profile
def index(num, data, H, C, parameter, poolSize):  # num in [0,1,..,.,poolsize-1]
    n = data.shape[0]
    encodeidea = parameter.encodeidea
    M = parameter.M
    batch = n // poolSize

    if num==poolSize:
        train_data = data[num * batch:,...]
        H = H[num * batch:, ...]
    else:            
        train_data = data[num * batch:(num + 1) * batch, ...]
        H = H[num * batch:(num + 1) * batch, ...]
    
    shape = train_data.shape[0] 
    Ind = np.zeros((shape, M), dtype=int)
    S_ind = np.zeros((shape, M), dtype=int)
    for i in range(shape):
        x = train_data[i, ...]
        weight = H[i, ...]

        ind_i,S_ind[i,...] = encodeidea(x, weight, C, parameter)

        Ind[i,...] = ind_i
    
    return Ind, S_ind


# @profile
# @timefn
def multiprocessing_index(data, H, C, parameter, poolSize):
    n = data.shape[0]
    batch = n // poolSize
    if n%poolSize==0:
        poolnum = poolSize
    else:
        poolnum = poolSize + 1

    Ind = np.zeros((n,parameter.M),dtype=int)

    S_ind = np.zeros((n,parameter.M), dtype = int) 

    p = Pool(poolSize)
    a = p.map_async(partial(index, data=data, H=H, C=C, parameter=parameter, poolSize=poolSize), range(poolnum)).get()
    
    for i in range(poolSize):
        Ind[i * batch:(i + 1) * batch, ...] = a[i][0]
        S_ind[i * batch:(i + 1) * batch, ...] = a[i][1]

    if n % poolSize != 0:
        Ind[poolSize * batch:,...] = a[poolSize][0]
        S_ind[poolSize * batch:,...] = a[poolSize][1]

    p.close()
    p.join()

    return Ind,S_ind
