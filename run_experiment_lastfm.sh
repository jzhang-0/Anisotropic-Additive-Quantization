#!/bin/bash

M=8
K=16
K_v=400
output_file=save/LastFM/
suffix=AQ_M_${M}_K_${K}


python script/main_vq.py\
    --data data/data_LastFM.npy\
    --D 32\
    --K_v ${K_v}\
    --T 0.01\
    --nor 0\
    --output_file ${output_file}

python script/main_aq.py\
    --mode IVFAQ\
    --data data/data_LastFM.npy\
    --D 32\
    --K_v ${K_v}\
    --T 0.05\
    --nor 0\
    --output_file ${output_file}\
    --M ${M}\
    --K ${K}\
    --inv_tab ${output_file}inversted_table.txt\
    --code_book ${output_file}code_book.npy

python script/main_aq.py\
    --mode Recall\
    --inv_tab ${output_file}inversted_table.txt\
    --code_book ${output_file}code_book.npy\
    --output_file ${output_file}\
    --queries data/queries_LastFM.npy\
    --tr100 data/true_neighbors_top100.npy\
    --aq_code ${output_file}aq_code_${suffix}.npy\
    --aq_codebooks ${output_file}codebook_C_${suffix}.npy\
    --num_to_search 1000\
    --topk 512
