import sys 
sys.path.append("src") 

import argparse
import os
import pickle
import numpy as np

from scann_VQ import ScaNN_VQ

parser = argparse.ArgumentParser(description='Initialize Parameters!')

parser.add_argument('--data', default='-1', type=str, help='path of datafile')
parser.add_argument('--D', default=-1, type=int, help='the dim of data')
parser.add_argument('--K_v', default=-1, type=int, help='the number of cluster centroids (num_leaves)') 
parser.add_argument('--T', default=-1, type=float, help='anisotropic_quantization_threshold=T*mean(norm)')
parser.add_argument('--nor', default=-1, type=int, help='Whether the data is normalized')
parser.add_argument('--output_file', default='-1', type=str, help='Directory path to save the result')

config = parser.parse_args()

def main_vq(config):
    config.suffix = f"VQ_{config.K_v}"

    dataset = np.load(config.data)
    print(f"{config.suffix},T:{config.T},data number:{dataset.shape[0]},normalization:{config.nor}")
    if config.nor == 0:
        norm = np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        config.T = config.T*np.mean(norm) 
    
    s_vq = ScaNN_VQ(config.D,config.T,config.nor,config.K_v)
    weight_H_set,_ = s_vq.compute_H_Direction(dataset)
    inversted_table, code_book, loss = s_vq.kmeans_ScaNN(dataset,weight_H_set)

    dirs = config.output_file
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    with open(dirs+"inversted_table.txt", "wb") as fp:
        pickle.dump(inversted_table,fp)
    np.save(dirs+"code_book",code_book)
    print("The Anisotropic VQ is complete.")

if __name__ == "__main__":
    main_vq(config)