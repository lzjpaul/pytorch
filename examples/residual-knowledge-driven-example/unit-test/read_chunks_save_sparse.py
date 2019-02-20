# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import sys
import argparse
import pandas as pd
import scipy.sparse


def read_csv_chunks(filename, chunksize):
    index = 0
    for gm_chunk in pd.read_csv(filename, header=None, index_col=None, chunksize=chunksize):
        print ("index: ", index)
        if index == 0:
            file_concat = gm_chunk.values
        else: 
            file_concat = np.concatenate((file_concat, gm_chunk.values), axis=0)
        print ("file_concat shape: ", file_concat.shape)
        index = index + 1
    print ("final file_concat dtype: ", file_concat.dtype)
    print ("final file_concat shape: ", file_concat.shape)
    return file_concat
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Sparse')
    parser.add_argument('-srcdatadir', type=str, help='source data directory')
    parser.add_argument('-dstdatadir', type=str, help='destination data directory')
    parser.add_argument('-chunksize', type=int, help='chunk size')
    args = parser.parse_args()

    print ("reading spurce data")
    print ("source data begins")
    source_x = read_csv_chunks(args.srcdatadir, args.chunksize)
    print ("source data ends")
    print ("source_x shape: ", source_x.shape)
    
    source_x_sparse_matrix = scipy.sparse.csr_matrix(source_x)
    # print ("source_x_sparse_matrix: ", source_x_sparse_matrix)
    scipy.sparse.save_npz(args.dstdatadir, source_x_sparse_matrix)
# python read_in_data_memory_pandas.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_x_seq.csv -trainlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname lstm
# python read_in_data_memory.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq.csv -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq.csv -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname lstm
