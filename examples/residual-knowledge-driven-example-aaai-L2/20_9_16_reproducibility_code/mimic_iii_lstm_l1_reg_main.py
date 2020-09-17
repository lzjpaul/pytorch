import torch
import torch.nn as nn
from torch.autograd import Variable
from init_linear import InitLinear
from res_regularizer import ResRegularizer
import torch.utils.data as Data
import torch.autograd as autograd
import random
import time
import math
import sys
import numpy as np
import scipy.sparse
import logging
import argparse
from torch.optim.adam import Adam
import torch.optim as optim
import datetime
import torch.nn.functional as F
from mimic_metric import *
import data
from baseline_method import BaselineMethod

features = []

class BasicRNNBlock(nn.Module):
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first):
        super(BasicRNNBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicRNNBlock')
        if self.gpu_id >= 0:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id))
        else:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
    ###
    ### 
    ## actually, this is useless of the input x is already organized according to batch_first
    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            rnn_out, self.hidden = self.rnn(
                x.view(batch_size, -1, self.input_dim), self.hidden)
        else:
            batch_size = x.size()[1]
            rnn_out, self.hidden = self.rnn(
                x.view(-1, batch_size, self.input_dim), self.hidden)
        logger.debug ('rnn_out size: ')
        logger.debug (rnn_out.data.size())
        logger.debug ('rnn_out norm: %f', rnn_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/x ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(x.data.cpu().numpy()))
        return rnn_out


class BasicDropoutRNNBlock(nn.Module):
    # no seq_length 
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first, dropout):
        super(BasicDropoutRNNBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.drop1 = nn.Dropout(dropout)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicRNNBlock')
        if self.gpu_id >= 0:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id))
        else:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
    ### 
    ### 
    ## actually, this is useless of the input x is already organized according to batch_first
    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            x = x.view(batch_size, -1, self.input_dim)
            x = self.drop1(x)
            features.append(x.data)
            logger.debug('Inside ' + self.__class__.__name__ + ' forward')
            logger.debug ('after dropout data size:')
            logger.debug (x.data.size())
            logger.debug ('after dropout data norm: %f', x.data.norm())
            rnn_out, self.hidden = self.rnn(
                x, self.hidden)
        else:
            batch_size = x.size()[1]
            x = x.view(-1, batch_size, self.input_dim)
            x = self.drop1(x)
            features.append(x.data)
            logger.debug('Inside ' + self.__class__.__name__ + ' forward')
            logger.debug ('after dropout data size:')
            logger.debug (x.data.size())
            logger.debug ('after dropout data norm: %f', x.data.norm())
            rnn_out, self.hidden = self.rnn(
                x, self.hidden)
        logger.debug ('rnn_out size: ')
        logger.debug (rnn_out.data.size())
        logger.debug ('rnn_out norm: %f', rnn_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/x ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(x.data.cpu().numpy()))
        return rnn_out

class BasicLSTMBlock(nn.Module):
    # no seq_length 
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first):
        super(BasicLSTMBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicRNNBlock')
        if self.gpu_id >= 0:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)), 
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)))
        else:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))
    ### 
    ### 
    ## sequence length is not fixed, so I do not pass in sequence_length
    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            lstm_out, self.hidden = self.lstm(
                x.view(batch_size, -1, self.input_dim), self.hidden)
        else:
            batch_size = x.size()[1]
            lstm_out, self.hidden = self.lstm(
                x.view(-1, batch_size, self.input_dim), self.hidden)
        logger.debug ('lstm_out size: ')
        logger.debug (lstm_out.data.size())
        logger.debug ('lstm_out norm: %f', lstm_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/x ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(x.data.cpu().numpy()))
        return lstm_out

class BasicDropoutLSTMBlock(nn.Module):
    #  
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first, dropout):
        super(BasicDropoutLSTMBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.drop1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicRNNBlock')
        if self.gpu_id >= 0:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)), 
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)))
        else:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))
    ### 
    ### 
    ## sequence length is not fixed, so I do not pass in sequence_length
    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            x = x.view(batch_size, -1, self.input_dim)
            x = self.drop1(x)
            features.append(x.data)
            logger.debug('Inside ' + self.__class__.__name__ + ' forward')
            logger.debug ('after dropout data size:')
            logger.debug (x.data.size())
            logger.debug ('after dropout data norm: %f', x.data.norm())
            lstm_out, self.hidden = self.lstm(
                x, self.hidden)
        else:
            batch_size = x.size()[1]
            x = x.view(-1, batch_size, self.input_dim)
            x = self.drop1(x)
            features.append(x.data)
            logger.debug('Inside ' + self.__class__.__name__ + ' forward')
            logger.debug ('after dropout data size:')
            logger.debug (x.data.size())
            logger.debug ('after dropout data norm: %f', x.data.norm())
            lstm_out, self.hidden = self.lstm(
                x, self.hidden)
        logger.debug ('lstm_out size: ')
        logger.debug (lstm_out.data.size())
        logger.debug ('lstm_out norm: %f', lstm_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/x ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(x.data.cpu().numpy()))
        return lstm_out

class BasicResRNNBlock(nn.Module):
    # 
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first):
        super(BasicResRNNBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicResRNNBlock')
        if self.gpu_id >= 0:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id))
        else:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
    ### 
    ### 
    ## actually, this is useless of the input x is already organized according to batch_first
    def forward(self, x):
        residual = x
        # print ('residual norm: ', np.linalg.norm(residual.data.cpu().numpy()))
        logger = logging.getLogger('res_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            rnn_out, self.hidden = self.rnn(
                x.view(batch_size, -1, self.input_dim), self.hidden)
            rnn_out = rnn_out + residual.view(batch_size, -1, self.input_dim)
        else:
            batch_size = x.size()[1]
            rnn_out, self.hidden = self.rnn(
                x.view(-1, batch_size, self.input_dim), self.hidden)
            rnn_out = rnn_out + residual.view(-1, batch_size, self.input_dim)
        logger.debug ('rnn_out size: ')
        logger.debug (rnn_out.data.size())
        logger.debug ('rnn_out norm: %f', rnn_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/residual ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(residual.data.cpu().numpy()))
        return rnn_out

class BasicResLSTMBlock(nn.Module):
    # 
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first):
        super(BasicResLSTMBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicResRNNBlock')
        if self.gpu_id >= 0:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)),
                    autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)))
        else:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))
    ### 
    ### 
    ## actually, this is useless of the input x is already organized according to batch_first
    def forward(self, x):
        residual = x
        # print ('residual norm: ', np.linalg.norm(residual.data.cpu().numpy()))
        logger = logging.getLogger('res_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            lstm_out, self.hidden = self.lstm(
                x.view(batch_size, -1, self.input_dim), self.hidden)
            lstm_out = lstm_out + residual.view(batch_size, -1, self.input_dim)
        else:
            batch_size = x.size()[1]
            lstm_out, self.hidden = self.lstm(
                x.view(-1, batch_size, self.input_dim), self.hidden)
            lstm_out = lstm_out + residual.view(-1, batch_size, self.input_dim)
        logger.debug ('lstm_out size: ')
        logger.debug (lstm_out.data.size())
        logger.debug ('lstm_out norm: %f', lstm_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/residual ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(residual.data.cpu().numpy()))
        return lstm_out

class ResNetRNN(nn.Module):
    # 
    def __init__(self, gpu_id, block, input_dim, hidden_dim, output_dim, blocks, batch_first):
        super(ResNetRNN, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_first = batch_first
        #for this one, I can use init_hidden to initialize hidden layer, also only return rnn_out
        self.rnn1 = BasicRNNBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first)
        self.fc1 = InitLinear(hidden_dim, output_dim)

        logger = logging.getLogger('res_reg')
        # 
        for idx, m in enumerate(self.modules()):
            # print ('idx and self.modules():')
            # print (idx)
            # print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            # print ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            # print ('init hidden m: ', m)
            if isinstance(m, BasicRNNBlock) or isinstance(m, BasicResRNNBlock):
                # print ('isinstance(m, BasicRNNBlock) or isinstance(m, BasicResRNNBlock)')
                m.hidden = m.init_hidden(batch_size)

    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first):
        logger = logging.getLogger('res_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # 
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.rnn1(x)
        features.append(x.data)
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('before blocks norm: %f', x.data.norm())
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        ## 
        if self.batch_first:
            x = F.sigmoid(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = F.sigmoid(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x

class ResNetDropoutRNN(nn.Module):
    # no seq_length 
    def __init__(self, gpu_id, block, input_dim, hidden_dim, output_dim, blocks, batch_first, dropout):
        super(ResNetDropoutRNN, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_first = batch_first
        #for this one, I can use init_hidden to initialize hidden layer, also only return rnn_out
        self.rnn1 = BasicRNNBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first, dropout)
        self.fc1 = InitLinear(hidden_dim, output_dim)

        logger = logging.getLogger('res_reg')
        # 
        for idx, m in enumerate(self.modules()):
            # print ('idx and self.modules():')
            # print (idx)
            # print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            # print ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            # print ('init hidden m: ', m)
            if isinstance(m, BasicRNNBlock) or isinstance(m, BasicResRNNBlock) or isinstance(m, BasicDropoutRNNBlock):
                # print ('isinstance(m, BasicRNNBlock) or isinstance(m, BasicResRNNBlock)')
                m.hidden = m.init_hidden(batch_size)

    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first, dropout):
        logger = logging.getLogger('res_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first, dropout))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first, dropout)) # 
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.rnn1(x)
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        ## 
        if self.batch_first:
            x = F.sigmoid(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = F.sigmoid(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x


class ResNetLSTM(nn.Module):
    # no seq_length 
    def __init__(self, gpu_id, block, input_dim, hidden_dim, output_dim, blocks, batch_first):
        super(ResNetLSTM, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_first = batch_first
        #for this one, I can use init_hidden to initialize hidden layer, also only return rnn_out
        self.lstm1 = BasicLSTMBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first)
        self.fc1 = InitLinear(hidden_dim, output_dim)

        logger = logging.getLogger('res_reg')
        # 
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock):
                logger.debug ('isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock)')
                m.hidden = m.init_hidden(batch_size)

    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first):
        logger = logging.getLogger('res_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # 
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        print ('layers: ')
        return nn.Sequential(*layers)

    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.lstm1(x)
        features.append(x.data)
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('before blocks norm: %f', x.data.norm())
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        ## 
        if self.batch_first:
            x = F.sigmoid(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = F.sigmoid(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x

class ResNetDropoutLSTM(nn.Module):
    # no seq_length 
    def __init__(self, gpu_id, block, input_dim, hidden_dim, output_dim, blocks, batch_first, dropout):
        super(ResNetDropoutLSTM, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_first = batch_first
        #for this one, I can use init_hidden to initialize hidden layer, also only return rnn_out
        self.lstm1 = BasicLSTMBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first, dropout)
        self.fc1 = InitLinear(hidden_dim, output_dim)

        logger = logging.getLogger('res_reg')
        # 
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock) or isinstance(m, BasicDropoutLSTMBlock):
                logger.debug ('isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock) or isinstance(m, BasicDropoutLSTMBlock)')
                m.hidden = m.init_hidden(batch_size)

    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first, dropout):
        logger = logging.getLogger('res_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first, dropout))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first, dropout)) #
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        print ('layers: ')
        return nn.Sequential(*layers)

    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.lstm1(x)
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        ## 
        if self.batch_first:
            x = F.sigmoid(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = F.sigmoid(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class WLMResNetLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, gpu_id, block, ntoken, input_dim, hidden_dim, blocks, batch_first, tie_weights=False):
        super(WLMResNetLSTM, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.encoder = nn.Embedding(ntoken, input_dim)
        self.lstm1 = BasicLSTMBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first)
        self.decoder = nn.Linear(hidden_dim, ntoken)
        
        logger = logging.getLogger('res_reg')
        # 
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_dim != input_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()


    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock):
                logger.debug ('isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock)')
                m.hidden = m.init_hidden(batch_size)

    def repackage_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock):
                logger.debug ('wlm repackage_hidden isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock)')
                # print ('before repackage m.hidden: ', m.hidden)
                m.hidden = repackage_hidden(m.hidden)
                # print ('after repackage m.hidden: ', m.hidden)
    
    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first):
        logger = logging.getLogger('res_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # 
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        print ('layers: ')
        return nn.Sequential(*layers)


    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.encoder(x)
        logger.debug('after encoder x shape')
        logger.debug (x.shape)
        x = self.lstm1(x)
        features.append(x.data)
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('before blocks norm: %f', x.data.norm())
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        decoded = self.decoder(x.view(x.size(0)*x.size(1), x.size(2)))
        return decoded.view(x.size(0), x.size(1), decoded.size(1))

class WLMResNetDropoutLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, gpu_id, block, ntoken, input_dim, hidden_dim, blocks, batch_first, dropout, tie_weights=False):
        super(WLMResNetDropoutLSTM, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.encoder = nn.Embedding(ntoken, input_dim)
        self.lstm1 = BasicLSTMBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first, dropout)
        self.decoder = nn.Linear(hidden_dim, ntoken)
        
        logger = logging.getLogger('res_reg')
        # 
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_dim != input_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()


    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock) or isinstance(m, BasicDropoutLSTMBlock):
                logger.debug ('isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock) or isinstance(m, BasicDropoutLSTMBlock)')
                m.hidden = m.init_hidden(batch_size)

    def repackage_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock) or isinstance(m, BasicDropoutLSTMBlock):
                logger.debug ('wlm repackage_hidden isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock) or isinstance(m, BasicDropoutLSTMBlock)')
                # print ('before repackage m.hidden: ', m.hidden)
                m.hidden = repackage_hidden(m.hidden)
                # print ('after repackage m.hidden: ', m.hidden)
    
    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first, dropout):
        logger = logging.getLogger('res_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first, dropout))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first, dropout)) # 
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        print ('layers: ')
        return nn.Sequential(*layers)


    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.encoder(x)
        logger.debug('after encoder x shape')
        logger.debug (x.shape)
        x = self.lstm1(x)
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        decoded = self.decoder(x.view(x.size(0)*x.size(1), x.size(2)))
        return decoded.view(x.size(0), x.size(1), decoded.size(1))


def get_features_hook(module, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    
    logger = logging.getLogger('res_reg')
    logger.debug('Inside ' + module.__class__.__name__ + ' forward hook')
    logger.debug('')
    # logger.debug('input:')
    # logger.debug(input)
    logger.debug('input: ')
    logger.debug(type(input))
    logger.debug('input[0]: ')
    logger.debug(type(input[0]))
    logger.debug('output: ')
    logger.debug(type(output))
    logger.debug('')
    logger.debug('input[0] size:')
    logger.debug(input[0].size())
    logger.debug('input norm: %f', input[0].data.norm())
    logger.debug('output size:')
    logger.debug(output.data.size())
    logger.debug('output norm: %f', output.data.norm())
    features.append(output.data)

def get_batch(source, i, seqnum):
    seq_len = min(seqnum, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchify(data, bsz, gpu_id):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print ("data shape: ", data.shape)
    return data.cuda(gpu_id)


def train(model_name, rnn, gpu_id, train_loader, test_loader, criterion, optimizer, reg_method, prior_beta, reg_lambda, momentum_mu, blocks, n_hidden, weightdecay, firstepochs, labelnum, batch_first, n_epochs, lasso_strength, max_val):
    logger = logging.getLogger('res_reg')
    res_regularizer_instance = ResRegularizer(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=blocks, feature_dim=n_hidden, model_name=model_name)
    baseline_method_instance = BaselineMethod()
    # Keep track of losses for plotting
    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    pre_running_loss = 0.0
    for epoch in range(n_epochs):
        rnn.train()
        running_loss = 0.0
        running_accuracy = 0.0
        # output, loss = train(model_name, epoch, batch_size, batch_first, category_tensor, line_tensor, res_regularizer_instance)
        for batch_idx, data_iter in enumerate(train_loader, 0):
            data_x, data_y = data_iter
            # print ('data_y shape: ', data_y.shape)
            data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
            rnn.init_hidden(data_x.shape[0])
            optimizer.zero_grad()
            features.clear()
            logger.debug ('data_x: ')
            logger.debug (data_x)
            logger.debug ('data_x norm: %f', data_x.norm())
            outputs = rnn(data_x)
            loss = criterion(outputs, data_y)
            logger.debug ("features length: %d", len(features))
            for feature in features:
                logger.debug ("feature size:")
                logger.debug (feature.data.size())
                logger.debug ("feature norm: %f", feature.data.norm())
            accuracy = AUCAccuracy(outputs.data.cpu().numpy(), data_y.data.cpu().numpy())[0]
            loss.backward()
            ### print norm
            if (epoch == 0 and batch_idx < 1000) or batch_idx % 1000 == 0:
                for name, f in rnn.named_parameters():
                    print ('batch_idx: ', batch_idx)
                    print ('param name: ', name)
                    print ('param size:', f.data.size())
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                    print ('lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### when to use res_reg
                
            if "reg" in model_name:
                if reg_method == 6 and epoch >= firstepochs:
                    feature_idx = -1 # which feature to use for regularization
                for name, f in rnn.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (f.size())
                    if "layer1" in name and "weight_ih" in name:
                        if reg_method == 6 and epoch >= firstepochs:  # corr-reg
                            logger.debug ('corr_reg param name: '+ name)
                            feature_idx = feature_idx + 1
                            cal_all_timesteps=False
                            res_regularizer_instance.apply(model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, 1, len(train_loader.dataset), epoch, f, name, batch_idx, batch_first, cal_all_timesteps)
                            # print ("check len(train_loader.dataset): ", len(train_loader.dataset))
                        elif reg_method == 7:  # L1-norm
                            logger.debug ('L1 norm param name: '+ name)
                            logger.debug ('lasso_strength: %f', lasso_strength)
                            baseline_method_instance.lasso_regularization(f, lasso_strength)
                        else:  # maxnorm and dropout
                            logger.debug ('no actions of param grad for maxnorm or dropout param name: '+ name)
                    else:
                        if weightdecay != 0:
                            logger.debug ('weightdecay name: ' + name)
                            logger.debug ('weightdecay: %f', weightdecay)
                            f.grad.data.add_(float(weightdecay), f.data)
                            logger.debug ('param norm: %f', np.linalg.norm(f.data.cpu().numpy()))
                            logger.debug ('weightdecay norm: %f', np.linalg.norm(float(weightdecay)*f.data.cpu().numpy()))
                            logger.debug ('lr 1.0 * param grad norm: %f', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### print norm
            optimizer.step()

            ### maxnorm constraist
            if "reg" in model_name and reg_method == 8:
                for name, param in rnn.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (param.size())
                    if "layer1" in name and "weight_ih" in name:
                        logger.debug ('max norm constraint for param name: '+ name)
                        logger.debug ('max_val: %f', max_val)
                        baseline_method_instance.max_norm(param, max_val)
            ### maxnorm constraist

            running_loss += loss.item() * len(data_x)
            # 
            running_accuracy += accuracy * len(data_x)

        # Print epoch number, loss, name and guess
        # print ('maximum batch_idx: ', batch_idx)
        running_loss = running_loss / len(train_loader.dataset)
        running_accuracy = running_accuracy / len(train_loader.dataset)
        print('epoch: %d, training loss per sample per label =  %f, training accuracy =  %f'%(epoch, running_loss, running_accuracy))
        print('abs(running_loss - pre_running_loss)', abs(running_loss - pre_running_loss))
        pre_running_loss = running_loss

        # test
        rnn.eval()
        outputs_list = []
        labels_list = []
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, data_iter in enumerate(test_loader):
                data_x, data_y = data_iter
                # print ("test data_y shape: ", data_y.shape)
                data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
                rnn.init_hidden(data_x.shape[0])
                outputs = rnn(data_x)
                # print ('outputs shape: ', outputs.shape)
                # print ('data_y shape: ', data_y.shape)
                loss = criterion(outputs, data_y)
                test_loss += loss.item() * len(data_x)
                outputs_list.extend(list(outputs.data.cpu().numpy()))
                labels_list.extend(list(data_y.data.cpu().numpy()))
            # print ('test outputs_list length: ', len(outputs_list))
            # print ('test labels_list length: ', len(labels_list))
            metrics = AUCAccuracy(np.array(outputs_list), np.array(labels_list))
            accuracy, macro_auc, micro_auc = metrics[0], metrics[1], metrics[2]
            # print ('maximum test batch_idx: ', batch_idx)
            test_loss = test_loss / len(test_loader.dataset)
            print ('test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(test_loss, accuracy, macro_auc, micro_auc))
            if epoch == (n_epochs - 1):
                print ('| final weightdecay {:.10f} | final prior_beta {:.10f} | final reg_lambda {:.10f} | final lasso_strength {:.10f} | final max_val {:.10f}'.format(weightdecay, prior_beta, reg_lambda, lasso_strength, max_val))
                print ('final test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(test_loss, accuracy, macro_auc, micro_auc))

    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print (do)
    elapsed = done - start
    print (elapsed)
    print('Finished Training')


def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print ("new param_group['lr']: ", param_group['lr'])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def trainwlm(model_name, rnn, gpu_id, corpus, batchsize, train_data, val_data, test_data, seqnum, clip, criterion, optimizer, reg_method, prior_beta, reg_lambda, momentum_mu, blocks, n_hidden, weightdecay, firstepochs, labelnum, batch_first, n_epochs, lasso_strength, max_val):
    logger = logging.getLogger('res_reg')
    res_regularizer_instance = ResRegularizer(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=blocks, feature_dim=n_hidden, model_name=model_name)
    baseline_method_instance = BaselineMethod()
    # Keep track of losses for plotting
    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    pre_running_loss = 0.0
    best_val_loss = None
    for epoch in range(n_epochs):
        rnn.train()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        rnn.init_hidden(batchsize) # since the data is batchfied, so batchsize can be just passed in from main()
        for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, seqnum)):
            data_x, data_y = get_batch(train_data, i, seqnum)
            # print ('data_y shape: ', data_y.shape)
            data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))

            rnn.repackage_hidden()
            optimizer.zero_grad()
            features.clear()
            logger.debug ('data_x: ')
            logger.debug (data_x)
            logger.debug ('data_x shape: ')
            logger.debug (data_x.shape)
            # logger.debug ('data_x norm: %f', data_x.norm())
            outputs = rnn(data_x)
            loss = criterion(outputs.view(-1, ntokens), data_y)
            logger.debug ("features length: %d", len(features))
            for feature in features:
                logger.debug ("feature size:")
                logger.debug (feature.data.size())
                logger.debug ("feature norm: %f", feature.data.norm())
            loss.backward()
            # print ("batch_idx: ", batch_idx)
            ### print norm
            if (epoch == 0 and batch_idx < 1000) or batch_idx % 1000 == 0:
                for name, f in rnn.named_parameters():
                    print ('batch_idx: ', batch_idx)
                    print ('param name: ', name)
                    print ('param size:', f.data.size())
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                    print ('lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### when to use res_reg
            if "reg" in model_name:
                if reg_method == 6 and epoch >= firstepochs:
                    feature_idx = -1 # which feature to use for regularization
                for name, f in rnn.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (f.size())
                    if "layer1" in name and "weight_ih" in name:
                        if reg_method == 6 and epoch >= firstepochs:  # corr-reg
                            logger.debug ('corr_reg param name: '+ name)
                            feature_idx = feature_idx + 1
                            cal_all_timesteps=True
                            res_regularizer_instance.apply(model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, seqnum, (train_data.size(0) * train_data.size(1))/seqnum, epoch, f, name, batch_idx, batch_first, cal_all_timesteps)
                            # print ("check len(train_loader.dataset): ", len(train_loader.dataset))
                        elif reg_method == 7:  # L1-norm
                            logger.debug ('L1 norm param name: '+ name)
                            logger.debug ('lasso_strength: %f', lasso_strength)
                            baseline_method_instance.lasso_regularization(f, lasso_strength)
                        else:  # maxnorm and dropout
                            logger.debug ('no actions of param grad for maxnorm or dropout param name: '+ name)
                    else:
                        if weightdecay != 0:
                            logger.debug ('weightdecay name: ' + name)
                            logger.debug ('weightdecay: %f', weightdecay)
                            f.grad.data.add_(float(weightdecay), f.data)
                            logger.debug ('param norm: %f', np.linalg.norm(f.data.cpu().numpy()))
                            logger.debug ('weightdecay norm: %f', np.linalg.norm(float(weightdecay)*f.data.cpu().numpy()))
                            logger.debug ('lr 1.0 * param grad norm: %f', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### print norm
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
            optimizer.step()

            ### maxnorm constraist
            if "reg" in model_name and reg_method == 8:
                for name, param in rnn.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (param.size())
                    if "layer1" in name and "weight_ih" in name:
                        logger.debug ('max norm constraint for param name: '+ name)
                        logger.debug ('max_val: %f', max_val)
                        baseline_method_instance.max_norm(param, max_val)
            ### maxnorm constraist

            total_loss += loss.item()
            # 

        # Print epoch number, loss, name and guess
        # print ('maximum batch_idx: ', batch_idx)
        # actually the last mini-batch may contain less time-steps!! originally, the train loss is printed every args.log_interval mini-batches
        # not totally correct, but just show some hints, then ok
        cur_loss = total_loss / (batch_idx+1)
        print('| epoch {:3d} | lr {:.8f} | {:5d} batches '
                    'loss per sample per timestep {:.8f} | ppl {:8.2f}'.format(
                epoch, get_lr(optimizer), batch_idx, cur_loss, math.exp(cur_loss)))
        print ('abs(cur_loss - pre_running_loss)', abs(cur_loss - pre_running_loss))
        pre_running_loss = cur_loss
        total_loss = 0

        # validation
        # Turn on evaluation mode which disables dropout.
        rnn.eval()
        total_val_loss = 0.
        ntokens = len(corpus.dictionary)
        print ('ntokens: ', ntokens)
        rnn.init_hidden(batchsize)
        with torch.no_grad():
            for batch_idx in range(0, val_data.size(0) - 1, seqnum):
                data_x, data_y = get_batch(val_data, batch_idx, seqnum)
                # print ("val data_y shape: ", data_y.shape)
                data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
                outputs = rnn(data_x)
                outputs_flat = outputs.view(-1, ntokens)
                # print ('outputs_flat shape: ', outputs_flat.shape)
                # print ('data_y shape: ', data_y.shape)
                # sum over timesteps, this is absolutely correct even if the last mini-batch is not equal lenght of timesteps
                total_val_loss += len(data_x) * criterion(outputs_flat, data_y).item()
                rnn.repackage_hidden()
        average_val_loss = total_val_loss / (len(val_data) - 1)
        print('=' * 89)
        print('| End of training | val loss {:.8f} | val ppl {:8.2f}'.format(average_val_loss, math.exp(average_val_loss)))
        print('=' * 89)
        if not best_val_loss or average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
        else:
            adjust_learning_rate(optimizer, float(1/4.0))
        
        
        # test
        # Turn on evaluation mode which disables dropout.
        rnn.eval()
        total_test_loss = 0.
        ntokens = len(corpus.dictionary)
        print ('ntokens: ', ntokens)
        rnn.init_hidden(batchsize)
        with torch.no_grad():
            for batch_idx in range(0, test_data.size(0) - 1, seqnum):
                data_x, data_y = get_batch(test_data, batch_idx, seqnum)
                # print ("test data_y shape: ", data_y.shape)
                data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
                outputs = rnn(data_x)
                outputs_flat = outputs.view(-1, ntokens)
                # print ('outputs_flat shape: ', outputs_flat.shape)
                # print ('data_y shape: ', data_y.shape)
                # sum over timesteps, this is absolutely correct even if the last mini-batch is not equal lenght of timesteps
                total_test_loss += len(data_x) * criterion(outputs_flat, data_y).item()
                rnn.repackage_hidden()
        average_test_loss = total_test_loss / (len(test_data) - 1)
        print('=' * 89)
        print('| End of training | test loss {:.8f} | test ppl {:8.2f}'.format(average_test_loss, math.exp(average_test_loss)))
        print('=' * 89)
        if epoch == (n_epochs - 1):
            print('=' * 89)
            print ('| final weightdecay {:.10f} | final prior_beta {:.10f} | final reg_lambda {:.10f} | final lasso_strength {:.10f} | final max_val {:.10f}'.format(weightdecay, prior_beta, reg_lambda, lasso_strength, max_val))
            print('| End of training | final test loss {:.8f} | final test ppl {:8.2f}'.format(average_test_loss, math.exp(average_test_loss)))
            print('=' * 89)
            

    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print (do)
    elapsed = done - start
    print (elapsed)
    print('Finished Training')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Residual LSTM')
    parser.add_argument('-traindatadir', type=str, help='training data directory, also the data dir for word language model')
    parser.add_argument('-trainlabeldir', type=str, help='training label directory')
    parser.add_argument('-testdatadir', type=str, help='test data directory')
    parser.add_argument('-testlabeldir', type=str, help='test label directory')
    parser.add_argument('-seqnum', type=int, help='sequence number')
    parser.add_argument('-modelname', type=str, help='resnetrnn or reslstm or rnn or lstm')
    parser.add_argument('-blocks', type=int, help='number of blocks')
    parser.add_argument('-lr', type=float, help='0.001 for MIMIC-III')
    parser.add_argument('-batchsize', type=int, help='batch_size')
    parser.add_argument('-regmethod', type=int, help='regmethod: : 6-corr-reg, 7-Lasso, 8-maxnorm, 9-dropout')
    parser.add_argument('-firstepochs', type=int, help='first epochs when no regularization is imposed')
    parser.add_argument('-considerlabelnum', type=int, help='just a reminder, need to consider label number because the loss is averaged across labels')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    # parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--batch_first', action='store_true')
    parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer, 200 for word language model, 128 for other datasets')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    print ("args.debug: ", args.debug)
    print ("wlm args.batch_first: ", args.batch_first)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger('res_reg')
    logger.info ('#################################')
    gpu_id = args.gpuid
    print('gpu_id: ', gpu_id)

    ######################################################################
    # Load Data
    # ---------
    #
    if "wikitext" not in args.traindatadir:
        print ("not word language model")
        print ("loading train_x")
        # train_x = np.genfromtxt(args.traindatadir, dtype=np.float32, delimiter=',')
        if "MIMIC-III" in args.traindatadir: 
            train_x_sparse_matrix = scipy.sparse.load_npz(args.traindatadir)
            train_x_sparse_matrix = train_x_sparse_matrix.astype(np.float32)
            train_x = np.array(train_x_sparse_matrix.todense())
        else:
            train_x = np.genfromtxt(args.traindatadir, dtype=np.float32, delimiter=',')
        
        train_y = np.genfromtxt(args.trainlabeldir, dtype=np.float32, delimiter=',')
        train_y = train_y.reshape((train_y.shape[0],-1))
        print ("loading test_x")
        # test_x = np.genfromtxt(args.testdatadir, dtype=np.float32, delimiter=',')
        if "MIMIC-III" in args.traindatadir:
            test_x_sparse_matrix = scipy.sparse.load_npz(args.testdatadir)
            test_x_sparse_matrix = test_x_sparse_matrix.astype(np.float32)
            test_x = np.array(test_x_sparse_matrix.todense())
        else:
            test_x = np.genfromtxt(args.testdatadir, dtype=np.float32, delimiter=',')
        
        test_y = np.genfromtxt(args.testlabeldir, dtype=np.float32, delimiter=',')
        test_y = test_y.reshape((test_y.shape[0],-1))
        train_x = train_x.reshape((train_x.shape[0], args.seqnum, -1))
        test_x = test_x.reshape((test_x.shape[0], args.seqnum, -1))
        print ('train_x.shape: ', train_x.shape)
        print ('test_x.shape: ', test_x.shape)
        print ('train_y.shape: ', train_y.shape)
        print ('test_y.shape: ', test_y.shape)
        input_dim = train_x.shape[-1]
        print ('check input_dim: ', input_dim)

        train_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        train_loader = Data.DataLoader(dataset=train_dataset,
                                       batch_size=args.batchsize,
                                       shuffle=True)
        print ('check len(train_dataset): ', len(train_dataset))
        test_dataset = Data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        test_loader = Data.DataLoader(dataset=test_dataset,
                                       batch_size=args.batchsize,
                                       shuffle=True)
        print ('check len(test_dataset): ', len(test_dataset))

        label_num = train_y.shape[1]
        print ("check label number: ", label_num)
    else:
        label_num = 1
        print ("wlm check label number : ", label_num)
        corpus = data.Corpus(args.traindatadir)
        train_data = batchify(corpus.train, args.batchsize, gpu_id)
        val_data = batchify(corpus.valid, args.batchsize, gpu_id)
        test_data = batchify(corpus.test, args.batchsize, gpu_id)
        ntokens = len(corpus.dictionary)
        input_dim = args.emsize

    ########## using for
    weightdecay_list = [0.0001]
    reglambda_list = [1.0]
    priorbeta_list = [1.0]
    lasso_strength_list = [1e-6]
    max_val_list = [3.0]

    for weightdecay in weightdecay_list:
        for reg_lambda in reglambda_list:
            for prior_beta in priorbeta_list:
                for lasso_strength in lasso_strength_list:
                    for max_val in max_val_list:
                        print ('weightdecay: ', weightdecay)
                        print ('reg_lambda: ', reg_lambda)
                        print ('priot prior_beta: ', prior_beta)
                        print ('lasso_strength: ', lasso_strength)
                        print ('max_val: ', max_val)
                        ########## using for
                        n_hidden = args.nhid
                        n_epochs = args.maxepoch

                        if "wikitext" not in args.traindatadir:
                            if "res" in args.modelname and "rnn" in args.modelname:
                                print ('check resrnn model')
                                rnn = ResNetRNN(args.gpuid, BasicResRNNBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
                                rnn = rnn.cuda(args.gpuid)
                            elif "res" in args.modelname and "lstm" in args.modelname:
                                print ('check reslstm model')
                                rnn = ResNetLSTM(args.gpuid, BasicResLSTMBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
                                rnn = rnn.cuda(args.gpuid)
                            elif "res" not in args.modelname and "rnn" in args.modelname:
                                if "dropout" not in args.modelname:
                                    print ('check rnn model')
                                    rnn = ResNetRNN(args.gpuid, BasicRNNBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
                                    rnn = rnn.cuda(args.gpuid)
                                else:
                                    print ('check dropoutrnn model')
                                    rnn = ResNetDropoutRNN(args.gpuid, BasicDropoutRNNBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first, args.dropout)
                                    rnn = rnn.cuda(args.gpuid)
                            elif "res" not in args.modelname and "lstm" in args.modelname:
                                if "dropout" not in args.modelname:
                                    print ('check lstm model')
                                    rnn = ResNetLSTM(args.gpuid, BasicLSTMBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
                                    rnn = rnn.cuda(args.gpuid)
                                else:
                                    print ('check dropoutlstm model')
                                    rnn = ResNetDropoutLSTM(args.gpuid, BasicDropoutLSTMBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first, args.dropout)
                                    rnn = rnn.cuda(args.gpuid)
                            else:
                                print("Invalid model name, exiting...")
                                exit()
                        else:
                            if "res" in args.modelname and "lstm" in args.modelname:
                                rnn = WLMResNetLSTM(args.gpuid, BasicResLSTMBlock, ntokens, input_dim, n_hidden, args.blocks, args.batch_first, tie_weights=args.tied)
                                rnn = rnn.cuda(args.gpuid)
                            elif "res" not in args.modelname and "lstm" in args.modelname:
                                if "dropout" not in args.modelname:
                                    print ('check wlmlstm model')
                                    rnn = WLMResNetLSTM(args.gpuid, BasicLSTMBlock, ntokens, input_dim, n_hidden, args.blocks, args.batch_first, tie_weights=args.tied)
                                    rnn = rnn.cuda(args.gpuid)
                                else:
                                    print ('check dropoutwlmlstm model')
                                    rnn = WLMResNetDropoutLSTM(args.gpuid, BasicDropoutLSTMBlock, ntokens, input_dim, n_hidden, args.blocks, args.batch_first, args.dropout, tie_weights=args.tied)
                                    rnn = rnn.cuda(args.gpuid)
                            else:
                                print("Invalid model name, exiting...")
                                exit()


                        if "reg" in args.modelname:
                            print ('optimizer without wd')
                            # optimizer = Adam(rnn.parameters(), lr=args.lr)
                            if "wikitext" not in args.traindatadir:
                                optimizer = optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9)
                            else:
                                optimizer = optim.SGD(rnn.parameters(), lr=args.lr)
                        else:
                            print ('optimizer with wd')
                            # optimizer = Adam(rnn.parameters(), lr=args.lr, weight_decay=args.decay)
                            if "wikitext" not in args.traindatadir:
                                optimizer = optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=weightdecay)
                            else:
                                optimizer = optim.SGD(rnn.parameters(), lr=args.lr, weight_decay=weightdecay)
                            # optimizer = optim.SGD(rnn.parameters(), lr=args.lr, weight_decay=args.decay)
                        print ('optimizer: ', optimizer)

                        if "wikitext" not in args.traindatadir:
                            criterion = nn.BCELoss()
                        else:
                            criterion = nn.CrossEntropyLoss()
                        print ('criterion: ', criterion)
                        momentum_mu = 0.9 # momentum mu
                        if "wikitext" not in args.traindatadir:
                            train(args.modelname, rnn, args.gpuid, train_loader, test_loader, criterion, optimizer, args.regmethod, prior_beta, reg_lambda, momentum_mu, args.blocks, n_hidden, weightdecay, args.firstepochs, label_num, args.batch_first, args.maxepoch, lasso_strength, max_val)
                        else:
                            trainwlm(args.modelname, rnn, args.gpuid, corpus, args.batchsize, train_data, val_data, test_data, args.seqnum, args.clip, criterion, optimizer, args.regmethod, prior_beta, reg_lambda, momentum_mu, args.blocks, n_hidden, weightdecay, args.firstepochs, label_num, args.batch_first, args.maxepoch, lasso_strength, max_val)

