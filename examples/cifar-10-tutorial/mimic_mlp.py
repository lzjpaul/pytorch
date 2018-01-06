# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from mimic_dataset import MIMICDataset
from mimic_metric import *
import datetime
import time
from init_linear import InitLinear
'''
(0) dataset is try_dataset
(1) print shape and norm of the parameters
'''
########################################################################
trainset = MIMICDataset(feature_csv_file='data-repository/train_x.csv', label_csv_file='data-repository/train_y.csv')
# trainset = MIMICDataset(feature_csv_file='data-repository/feature_matrix_try.csv', label_csv_file='data-repository/result_matrix_try.csv')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
testset = MIMICDataset(feature_csv_file='data-repository/test_x.csv', label_csv_file='data-repository/test_y.csv')
# testset = MIMICDataset(feature_csv_file='data-repository/feature_matrix_try.csv', label_csv_file='data-repository/result_matrix_try.csv')
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
print ('test bath_size: ',len(testset))
########################################################################
# Let us show some of the training images, for fun.

import numpy as np

gpu_id = 1

# get some random training images
dataiter = iter(trainloader)
features, labels = dataiter.next()

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, feature_dim):
        super(Net, self).__init__()
        self.fc1 = InitLinear(feature_dim, 128)
        self.fc2 = InitLinear(128, 80)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

feature_dim = trainset.feature_dim()
label_dim = trainset.label_dim()
print ('feature_dim: ',(feature_dim))
print ('label_dim: ',(label_dim))
net = Net(feature_dim)
for f in net.parameters():
    print ('param size:', f.data.size())
    print ('param norm: ', torch.norm(f.data, 2))
net.cuda(gpu_id)



import torch.optim as optim

criterion = nn.BCELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=float(0.001*label_dim))

print('Beginning Training')

start = time.time()
st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
print (st)

max_epoch = 2000
# training
for epoch in range(max_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    running_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        features, labels = data['features'], data['label']
        features, labels = Variable(features.cuda(gpu_id)), Variable(labels.cuda(gpu_id))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(features)
        loss = criterion(outputs, labels)
        accuracy = AUCAccuracy(outputs, labels)[0]
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
        ## print norm
        optimizer.step()
        # print statistics
        running_loss += loss.data[0]
        running_accuracy += accuracy
    print ('maximum i: ', i)
    print('epoch: %d, training loss =  %f, training accuracy =  %f'%(epoch, running_loss / (i+1), running_accuracy / (i+1)))

    # test
    for data in testloader:
        features, labels = data['features'], data['label']
        features, labels = Variable(features.cuda(gpu_id)), Variable(labels.cuda(gpu_id))
        outputs = net(features)
        ## this has no backward??
        loss = criterion(outputs, labels)
        metrics = AUCAccuracy(outputs, labels)
        accuracy, macro_auc, micro_auc = metrics[0], metrics[1], metrics[2]
        print ('test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(loss.data[0], accuracy, macro_auc, micro_auc))
        if epoch == (max_epoch - 1):
            print ('final test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(loss.data[0], accuracy, macro_auc, micro_auc))

done = time.time()
do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
print (do)
elapsed = done - start
print (elapsed)
print('Finished Training')
