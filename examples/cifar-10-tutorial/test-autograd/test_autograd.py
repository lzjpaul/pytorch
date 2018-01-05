# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from mimic_dataset import MIMICDataset
# from mimic_metric import *
import datetime
import time
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

trainset = MIMICDataset(feature_csv_file='test_grad/train_x.csv', label_csv_file='test_grad/train_y.csv')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
print ('train bath_size: %d', len(trainset))
########################################################################
# Let us show some of the training images, for fun.

import numpy as np

gpu_id = 1


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, feature_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 80)

    def forward(self, x):
        # x = F.sigmoid(x)
        x = x * x * 2
        return x

feature_dim = trainset.feature_dim()
print ('feature_dim: %d', (feature_dim))
net = Net(feature_dim)
net.cuda(gpu_id)

import torch.optim as optim

criterion = nn.BCELoss(size_average=False)
# criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.001)

print('Beginning Training')

start = time.time()
st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
print (st)

w = Variable(torch.ones(8, 8).type(torch.DoubleTensor).cuda(gpu_id), requires_grad=True)

max_epoch = 1
# training
for epoch in range(max_epoch):  # loop over the dataset multiple times
    # running_loss = 0.0
    # running_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        features, labels = data['features'], data['label']
        # print (features)
        # print (labels)
        features, labels = Variable(features.type(torch.DoubleTensor).cuda(gpu_id)), Variable(labels.cuda(gpu_id))
        y_pred = F.sigmoid(features.mm(w))
        # zero the parameter gradients
        # optimizer.zero_grad()
        # forward + backward + optimize
        
        # outputs = net(features)
        loss = criterion(y_pred, labels)
        # loss = (y_pred - labels).pow(2).sum()
        # loss = (y_pred - labels).pow(2)
        # accuracy = AUCAccuracy(outputs, labels)[0]
        loss.backward()
        print (w.grad.data)
        # optimizer.step()
        # print statistics
        # running_loss += loss.data[0]
        # running_accuracy += accuracy
        # print ('i: %d', i)
    # print ('maximum i: %d', i)
    # print('epoch: %d, training loss =  %f, training accuracy =  %f' % (epoch, running_loss / i, running_accuracy / i))
    '''
    # test
    for data in testloader:
        features, labels = data
        outputs = net(Variable(features.cuda(gpu_id)))
        ## this has no backward??
        loss = criterion(outputs, labels)
        metrics = AUCAccuracy(outputs, labels)
        accuracy, macro_auc, micro_auc = metrics[0], metrics[1], metrics[2]
        print 'test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f' % (loss.data[0], accuracy, macro_auc, micro_auc)
        if epoch == (max_epoch - 1):
            print 'final test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f' % (loss.data[0], accuracy, macro_auc, micro_auc)

done = time.time()
do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
print do
elapsed = done - start
print elapsed
print('Finished Training')
'''
