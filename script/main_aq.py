import sys
sys.path.append("src")
import pickle
import os
import argparse

import numpy as np
from scannAQ import ScannAQ

from encoding import CoordinateDes, beam_search
from scannAQ_function import codebook_init

from utils import residual_set
from recall_IVFADC import Recall_IVFADC_AQ


parser = argparse.ArgumentParser(description='Initialize Parameters!')


parser.add_argument('--mode', default='-1', type=str, help='IVFAQ or Recall')

parser.add_argument('--data', default='-1', type=str, help='path of datafile')
parser.add_argument('--D', default=-1, type=int, help='the dim of data')
parser.add_argument('--K_v', default=-1, type=int, help='the number of cluster centroids (num_leaves)') 
parser.add_argument('--T', default=-1, type=float, help='anisotropic_quantization_threshold=T*mean(norm)')
parser.add_argument('--nor', default=-1, type=int, help='Whether the data is normalized')
parser.add_argument('--output_file', default='-1', type=str, help='Directory path to save the result')
parser.add_argument('--inv_tab', default='-1', type=str, help='path of inv_tab')
parser.add_argument('--code_book', default='-1', type=str, help='path of code_book file')

# IVFAQ
parser.add_argument('--M', default=-1, type=int, help='the number of codebooks') 
parser.add_argument('--K', default=-1, type=int, help='the number of codewords in each codebook')
parser.add_argument('--iter_num', default=3, type=int, help='the number of coordinate descent rounds')
parser.add_argument('--poolSize', default = os.cpu_count()//2, type=int) 
parser.add_argument('--N', default=16, type=int, help='the beam size of beam search')

# Recall
parser.add_argument('--queries', default='-1', type=str, help='path of queriesfile')
parser.add_argument('--tr100', default='-1', type=str, help='path of grountruth file')
parser.add_argument('--aq_code', default='-1', type=str, help='path of data aq_codes file')
parser.add_argument('--aq_codebooks', default='-1', type=str, help='path of aq codebooks file')
parser.add_argument('--num_to_search', default='-1', type=int, help='the number of search items')
parser.add_argument('--topk', default='512', type=int, help='top k neighbors return by algorithm')

parameter = parser.parse_args()
parameter.encodeidea = CoordinateDes  # CoordinateDes or beam_search


def main_aq_ivf(parameter):
    parameter.suffix = f"AQ_M_{parameter.M}_K_{parameter.K}"
    parameter.bits = int(parameter.M * np.log2(parameter.K))

    M = parameter.M
    K = parameter.K
    encodeidea = parameter.encodeidea

    print(f"normalization={parameter.nor}")
    print(f"bits={parameter.bits}")
    print(f"encodeidea={encodeidea.__name__}")
    print(f"M={M},K={K},T={parameter.T}*mean(norm)")

    dataset = np.load(parameter.data)
    parameter.D = dataset.shape[1]

    code_book = np.load(parameter.code_book)
    with open(parameter.inv_tab, "rb") as fp:
        inv_tab = pickle.load(fp)

    residualset = residual_set(dataset,inv_tab,code_book)

    C = codebook_init(residualset, M, K)
    norm = np.linalg.norm(residualset, axis=1)[:, np.newaxis]  
    parameter.norm = norm.mean()
    if parameter.nor:
        parameter.eta = (parameter.D - 1) * parameter.T ** 2 / (1 - parameter.T ** 2)


    train_data = residualset

    scann_aq = ScannAQ(train_data, parameter)
    C, Ind_Matrix, S_ind_Matrix = scann_aq.codebook_train(train_data, C, parameter, maxiter = 10)

    dirs = parameter.output_file
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.save( dirs + f"codebook_C_{parameter.suffix}", C)
    np.save(dirs + f"aq_code_{parameter.suffix}", Ind_Matrix)
    print("The Anisotropic AQ is complete.")

def main_recall(parameter):
    code_book = np.load(parameter.code_book)
    with open(parameter.inv_tab, "rb") as fp:
        inv_tab = pickle.load(fp)

    queries = np.load(parameter.queries)
    tr100 = np.load(parameter.tr100)

    rk =(1,2,4,8,10,16,20,32,64,100,128,256,512)
    recall_aq_ivf = Recall_IVFADC_AQ(inv_tab=inv_tab,code_book=code_book,queries=queries)

    C = np.load(parameter.aq_codebooks)
    aq_code = np.load(parameter.aq_code)

    print(f"the number of queries:{queries.shape[0]}")
    neighbors = recall_aq_ivf.search_neightbors_IVFADC_fixnum(C, aq_code, queries, num_to_search=parameter.num_to_search, topk=parameter.topk)
    dirs = parameter.output_file
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.save( dirs + "neighbors", neighbors)
    
    true_neighbors = tr100[:,0]

    for i in rk:
        recall = recall_aq_ivf.compute_recall(neighbors[...,-i:],true_neighbors)
        print(f"recall 1@{i} = {recall}")

    tr10 = tr100[:,0:10]
    rk =(10,16,20,32,64,100,128,256,512)
    for i in rk:
        recall = recall_aq_ivf.compute_recall(neighbors[...,-i:], tr10)
        print(f"recall 10@{i} = {recall}")


if __name__ == "__main__":
    if parameter.mode == "IVFAQ":
        main_aq_ivf(parameter)
    elif parameter.mode == "Recall":
        main_recall(parameter)
    else:
        parser.print_help()