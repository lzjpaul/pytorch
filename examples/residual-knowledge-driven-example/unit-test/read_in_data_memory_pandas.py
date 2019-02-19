import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import sys
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Input')
    parser.add_argument('-traindatadir', type=str, help='train data directory')
    parser.add_argument('-trainlabeldir', type=str, help='train label directory')
    parser.add_argument('-testdatadir', type=str, help='test data directory')
    parser.add_argument('-testlabeldir', type=str, help='test label directory')
    parser.add_argument('-seqnum', type=int, help='sequence number')
    parser.add_argument('-modelname', type=str, help='model name')
    args = parser.parse_args()

    print ("reading data")
    print ("train data begins")
    train_x = pd.read_csv(args.traindatadir, header=None, index_col=None)
    print ("train data ends")
    print ("train label begins")
    train_y = pd.read_csv(args.trainlabeldir, header=None, index_col=None)
    print ("train label ends")
    print ("train data reshapes begins")
    train_x = train_x.reshape((train_x.shape[0], args.seqnum, -1))
    print ("train data reshape ends")
    print ("test data begins")
    test_x = pd.read_csv(args.testdatadir, header=None, index_col=None)
    print ("test data ends")
    print ("test label begins")
    test_y = pd.read_csv(args.testlabeldir, header=None, index_col=None)
    print ("test labels ends")
    print ("test data reshape begins")
    test_x = test_x.reshape((test_x.shape[0], args.seqnum, -1))
    print ("test data reshape ends")
    print ("train_x shape: ", train_x.shape)
    print ("test_x shape: ", test_x.shape)

    if 'mlp' in args.modelname:
        train_x = np.sum(train_x, axis=1, keepdims=False)
        print ('train_x shape: ', train_x.shape)
        test_x = np.sum(test_x, axis=1, keepdims=False)
        print ('test_x shape: ', test_x.shape)

    train_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=100,
                                   shuffle=True)
    print ('len(train_dataset): ', len(train_dataset))
 
    test_dataset = Data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_loader = Data.DataLoader(dataset=test_dataset,
                                   batch_size=100,
                                   shuffle=True)
    print ('len(test_dataset): ', len(test_dataset))

    for i, data_iter in enumerate(train_loader, 0):
        data_x, data_y = data_iter
        print ('train_data_x size: ', data_x.size())
        print ('train_data_y size: ', data_y.size())

    for i, data_iter in enumerate(test_loader, 0):
        data_x, data_y = data_iter
        print ('test_data_x size: ', data_x.size())
        print ('test_data_y size: ', data_y.size())
# python read_in_data_memory_pandas.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_x_seq.csv -trainlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname lstm
# python read_in_data_memory.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq.csv -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq.csv -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname lstm
