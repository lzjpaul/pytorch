# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from mimic_dataset import MIMICDataset, MIMICLSTMDataset 
from mimic_metric import *
import datetime
import time
from init_linear import InitLinear
from lda_regularizer import LDARegularizer
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.autograd as autograd

class LSTMNet(nn.Module):
    def __init__(self, gpu_id, batch_size, seq_lenth, feature_dim, hidden_dim, label_dim):
        super(LSTMNet, self).__init__()
        self.gpu_id = gpu_id
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_lenth = seq_lenth
        self.feature_dim = feature_dim
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, label_dim)
        # self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.gpu_id >= 0:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)))
        else:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))
    # inputs: (batch_size, seq, feature)
    def forward(self, inputs):
        lstm_out, self.hidden = self.lstm(
            inputs.view(-1, self.seq_lenth, self.feature_dim), self.hidden)
        x = F.sigmoid(self.fc1(lstm_out[:, -1, :]))
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pytorch mlp for healthcare dataset')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    parser.add_argument('-topicnum', type=int, help='topic_number')
    parser.add_argument('-hiddendim', type=int, help='hidden dimension')
    parser.add_argument('-timepoint', type=int, help='number of time points')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-weightdecay', type=float, help='weight decay value')
    parser.add_argument('-ldauptfreq', type=int, help='lda update frequency, in steps')
    parser.add_argument('-paramuptfreq', type=int, help='parameter update frequency, in steps')
    parser.add_argument('-batchsize', type=int, help='batchsize')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    parser.add_argument('-trainxpath', type=str, help='trainx path')
    parser.add_argument('-trainypath', type=str, help='trainy path')
    parser.add_argument('-testxpath', type=str, help='testx path')
    parser.add_argument('-testypath', type=str, help='testy path')
    parser.add_argument('-phipath', type=str, help='phi path')
    #parser.add_argument('-resultpath', type=str, help='result path')
    args = parser.parse_args()

    ########################################################################
    # trainset = MIMICLSTMDataset(feature_csv_file='data-repository/train_x.csv', label_csv_file='data-repository/train_y.csv', timepoint=args.timepoint)
    # trainset = MIMICLSTMDataset(feature_csv_file='sequence_data_repository/try_x_seq_100.csv', label_csv_file='sequence_data_repository/try_y_seq_100.csv', timepoint=args.timepoint)
    trainset = MIMICLSTMDataset(feature_csv_file=args.trainxpath, label_csv_file=args.trainypath, timepoint=args.timepoint)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    # testset = MIMICLSTMDataset(feature_csv_file='data-repository/test_x.csv', label_csv_file='data-repository/test_y.csv', timepoint=args.timepoint)
    # testset = MIMICLSTMDataset(feature_csv_file='sequence_data_repository/try_x_seq_100.csv', label_csv_file='sequence_data_repository/try_y_seq_100.csv', timepoint=args.timepoint)
    testset = MIMICLSTMDataset(feature_csv_file=args.testxpath, label_csv_file=args.testypath, timepoint=args.timepoint)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=True)
    ########################################################################
    # Let us show some of the training images, for fun.


    gpu_id = args.gpuid
    print ('gpu_id', gpu_id)
    
    print ('weight_decay', args.weightdecay)
    print ('learning rate', args.lr)
    # get some random training images
    dataiter = iter(trainloader)
    features, labels = dataiter.next()

    feature_dim = trainset.feature_dim()
    label_dim = trainset.label_dim()
    print ('feature_dim: ',(feature_dim))
    print ('label_dim: ',(label_dim))
    net = LSTMNet(gpu_id=gpu_id, batch_size=args.batchsize, seq_lenth=args.timepoint, feature_dim=feature_dim, hidden_dim=args.hiddendim, label_dim=label_dim)
    for name, param in net.named_parameters():
        print ('param name: ', name)
        print ('param size:', param.data.size())
        print ('param norm: ', torch.norm(param.data, 2))
    if gpu_id >= 0:
        net.cuda(gpu_id)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    # hyper parameters
    alpha = 1 + 0.05
    hyperpara = [alpha]
    doc_num = 4 * args.hiddendim # hard-code, number of hidden units 
    topic_num = args.topicnum
    word_num = feature_dim
    ldapara = [doc_num, topic_num, word_num]
    theta = [1.0/ldapara[1] for _ in range(ldapara[1])]
    phi = np.genfromtxt(args.phipath, delimiter=',')
    phi = np.transpose(phi)
    uptfreq = [args.ldauptfreq, args.paramuptfreq]
    lda_regularizer_instance = LDARegularizer(hyperpara=hyperpara, ldapara=ldapara, theta=theta, phi=phi, uptfreq=uptfreq)
    # hyper parameters
    print('Beginning Training')

    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print (st)

    max_epoch = args.maxepoch
    # training
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(trainloader, 0):
            features, labels = data['features'], data['label']
            if gpu_id >= 0:
                features, labels = Variable(features.cuda(gpu_id)), Variable(labels.cuda(gpu_id))
            else:
                features, labels = Variable(features), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Also, we need to clear out the hidden state of the LSTM.
            net.hidden = net.init_hidden(features.shape[0])
            # forward + backward + optimize
            outputs = net(features)
            loss = criterion(outputs, labels)
            accuracy = AUCAccuracy(outputs.data.cpu().numpy(), labels.data.cpu().numpy())[0]
            loss.backward()
            ### print norm
            '''
            for f in net.parameters():
                f.grad.data.div_(float(1./label_dim))
            
            for f in net.parameters():
                print ('param size:', f.data.size())
                print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                print ('0.001 * label_dim * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 0.001 * label_dim))
            '''
            # param name:  lstm.weight_ih_l0
            # param size:  torch.Size([24, 6])
            for name, param in net.named_parameters():
                # print ("param name: ", name)
                # print ("param size: ", param.size())
                # print ("")
                # print ("trainset number: ", len(trainset))
                # if name == 'weight':
                if name == 'lstm.weight_ih_l0':
                    lda_regularizer_instance.apply(gpu_id, len(trainset), label_dim, epoch, param, name, i)
                else:
                    if args.weightdecay != 0:
                        param.grad.data.add_(float(args.weightdecay), param.data)
            ## print norm
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            running_accuracy += accuracy
        print ('maximum i: ', i)
        print('epoch: %d, training loss =  %f, training accuracy =  %f'%(epoch, running_loss / (i+1), running_accuracy / (i+1)))

        # test
        outputs_list = []
        labels_list = []
        for data in testloader:
            features, labels = data['features'], data['label']
            if gpu_id >= 0:
                features, labels = Variable(features.cuda(gpu_id)), Variable(labels.cuda(gpu_id))
            else:
                features, labels = Variable(features), Variable(labels)
            net.hidden = net.init_hidden(features.shape[0])
            outputs = net(features)
            ## this has no backward??
            loss = criterion(outputs, labels)
            outputs_list.extend(list(outputs.data.cpu().numpy()))
            labels_list.extend(list(labels.data.cpu().numpy()))
        print ('test outputs_list length: ', len(outputs_list))
        print ('test labels_list length: ', len(labels_list))
        metrics = AUCAccuracy(np.array(outputs_list), np.array(labels_list))
        accuracy, macro_auc, micro_auc = metrics[0], metrics[1], metrics[2]
        print ('test loss (last minibatch) = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(loss.data[0], accuracy, macro_auc, micro_auc))
        if epoch == (max_epoch - 1):
            print ('final test loss (last minibatch) = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(loss.data[0], accuracy, macro_auc, micro_auc))

    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print (do)
    elapsed = done - start
    print (elapsed)
    print('Finished Training')
