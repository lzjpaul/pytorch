import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import sys
import argparse


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
    train_x = np.genfromtxt(args.traindatadir, delimiter=',')
    train_y = np.genfromtxt(args.trainlabeldir, delimiter=',')
    train_x = train_x.reshape((train_x.shape[0], args.seqnum, -1))
    test_x = np.genfromtxt(args.testdatadir, delimiter=',')
    test_y = np.genfromtxt(args.testlabeldir, delimiter=',')
    test_x = test_x.reshape((test_x.shape[0], args.seqnum, -1))
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
