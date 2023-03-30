import sys 
sys.path.append("src") 

import argparse
import os
import pickle
import numpy as np
from recall_IVFADC import Recall_IVFADC_PQ

from utils import codebook_init, residual_set
from scann_IVFADC import ScaNN_PQ_IVFADC

parser = argparse.ArgumentParser(description='Initialize Parameters!')

parser.add_argument('--mode', default='-1', type=str, help='IVFPQ or Recall')

parser.add_argument('--data', default='-1', type=str, help='path of datafile')
parser.add_argument('--D', default=-1, type=int, help='the dim of data')
parser.add_argument('--K_v', default=-1, type=int, help='the number of cluster centroids (num_leaves)') 
parser.add_argument('--T', default=-1, type=float, help='anisotropic_quantization_threshold=T*mean(norm)')
parser.add_argument('--nor', default=-1, type=int, help='Whether the data is normalized')
parser.add_argument('--output_file', default='-1', type=str, help='Directory path to save the result')
parser.add_argument('--inv_tab', default='-1', type=str, help='path of inv_tab')
parser.add_argument('--code_book', default='-1', type=str, help='path of code_book file')

# IVFPQ
parser.add_argument('--M', default=-1, type=int, help='the number of codebook') 
parser.add_argument('--K', default=-1, type=int, help='the number of codeword in each codebook')

# Recall
parser.add_argument('--queries', default='-1', type=str, help='path of queriesfile')
parser.add_argument('--tr100', default='-1', type=str, help='path of grountruth file')
parser.add_argument('--pq_code', default='-1', type=str, help='path of data codes file')
parser.add_argument('--pq_codebooks', default='-1', type=str, help='path of pq codebooks file')
parser.add_argument('--num_to_search', default='-1', type=int, help='the number of search items')
parser.add_argument('--topk', default='512', type=int, help='top k neighbors return by algorithm')

config = parser.parse_args()

def main_pq_ivf(config):
    config.suffix = f"VQ_{config.K_v}_PQ_M_{config.M}_K_{config.K}"

    dataset = np.load(config.data)
    code_book = np.load(config.code_book)
    with open(config.inv_tab, "rb") as fp:
        inv_tab = pickle.load(fp)

    print(f"{config.suffix},bits:{config.M*np.log2(config.K)},T:{config.T},normalization:{config.nor},train number:{dataset.shae[0]}")
    residualset = residual_set(dataset,inv_tab,code_book)

    if config.nor == 0:
        norm = np.linalg.norm(residualset, axis=1)[:, np.newaxis]
        config.T = config.T*np.mean(norm) 
        
    s_pq_inv = ScaNN_PQ_IVFADC(config.M,config.K,config.D,config.T,config.nor)
    C = codebook_init(residualset, config.M, config.K)

    C, S_ind, _ = s_pq_inv.train(train_data = residualset, C = C, saveTR=0)

    dirs = config.output_file
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    np.save(dirs+f"codebook_C_{config.suffix}",C)
    np.save(dirs+f"data_pqcodes_{config.suffix}",S_ind)

    print("The Anisotropic PQ is complete.")

def main_recall(config):
    queries = np.load(config.queries)

    code_book = np.load(config.code_book)
    with open(config.inv_tab, "rb") as fp:
        inv_tab = pickle.load(fp)

    C = np.load(config.pq_codebooks)
    code = np.load(config.pq_code)
    tr100 = np.load(config.tr100)

    recall_pq_ivf = Recall_IVFADC_PQ(M = config.M, Ks=config.K, D=config.D, inv_tab=inv_tab,code_book=code_book,queries=queries)


    Neighbor_Matrix = recall_pq_ivf.search_neightbors_IVFADC_fixnum(C,code,queries,num_to_search=config.num_to_search,topk=config.topk)
    true_neighbors = tr100[:,0]
    rk =(1,2,4,8,10,16,20,32,64,100,128,256,512)
    for i in rk:
        Neighbor_M = Neighbor_Matrix[...,-i:]
        result = recall_pq_ivf.compute_recall(Neighbor_M,true_neighbors)
        print(f"recall@{i} = {result}")

    tr10 = tr100[:,0:10]
    rk =(10,16,20,32,64,100,128,256,512)
    for i in rk:
        Neighbor_M = Neighbor_Matrix[...,-i:]
        result = recall_pq_ivf.compute_recall(Neighbor_M,tr10)
        print(f"recall 10@{i} = {result}")

if __name__ == "__main__":
    if config.mode == "IVFPQ":
        main_pq_ivf(config)
    elif config.mode == "Recall":
        main_recall(config)
    else:
        parser.print_help()
