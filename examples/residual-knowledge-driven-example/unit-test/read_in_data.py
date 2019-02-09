import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Input')
    parser.add_argument('-datadir', type=str, help='data directory')
    parser.add_argument('-labeldir', type=str, help='label directory')
    parser.add_argument('-seqnum', type=int, help='sequence number')
    parser.add_argument('-modelname', type=str, help='model name')
    args = parser.parse_args()

    train_x = np.genfromtxt(args.datadir, delimiter=',')
    train_y = np.genfromtxt(args.labeldir, delimiter=',')
    train_x = train_x.reshape((train_x.shape[0], args.seqnum, -1))
    print ("train_x shape: ", train_x.shape)

    if 'mlp' in args.modelname:
        train_x = np.sum(train_x, axis=1, keepdims=False)
        print ('train_x shape: ', train_x.shape)

    train_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=100,
                                   shuffle=True)
    print ('len(train_dataset): ', len(train_dataset))

    for i, data_iter in enumerate(train_loader, 0):
        if i > 10:
            break
        data_x, data_y = data_iter
        print ('data_x size: ', data_x.size())
        print ('data_y size: ', data_y.size())
