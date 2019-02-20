## https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
## refer to
## https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
## https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
## https://github.com/lzjpaul/pytorch/blob/LDA-regularization/examples/cifar-10-tutorial/mimic_lstm_lda.py
## https://github.com/lzjpaul/pytorch/blob/residual-knowledge-driven/examples/residual-knowledge-driven-example/mlp_residual.py
## https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py

## different from MIMIC-III
## (1) no seq_length
## (2) !! batch_first not True nned to modify many!!! 
##     --> forward function in BasicRNNBlock !! input and output are modified!!
##     --> forward function in ResNetRNN !! input and output and F.sigmoid(self.fc1(x[:, -1, :])) need to be modified!!
##     --> judge batch_first very slow ??
## (3) softmax loss ???


## Attention:
## (1) batch_first set as argument --> for calculating correlation also!!
## (2) label_num
## (3) batch_size are passed to the functions
## (4) different model different params???
## (5) sequence length is not fixed, so I do not pass in sequence_length + init_hidden need to pass in features[0] as batch_size + forward() batchsize need to be 
##     calculated while init_hidden batchsize is passed in by features[0]

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


features = []

class BasicRNNBlock(nn.Module):
    # no seq_length ?? batch_fist not true ??
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
    ### ??? !! batch_first: inputs.view(batch_size, -1, self.input_dim), self.hidden)
    ### ??? !! rnn_out return [:, -1, :]
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

class BasicLSTMBlock(nn.Module):
    # no seq_length ?? batch_fist not true ??
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
    ### ??? !! batch_first: inputs.view(batch_size, -1, self.input_dim), self.hidden)
    ### ??? !! rnn_out return [:, -1, :]
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

class BasicResRNNBlock(nn.Module):
    # no seq_length ?? batch_fist not true ??
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
    ### ??? !! batch_first: inputs.view(batch_size, -1, self.input_dim), self.hidden)
    ### ??? !! rnn_out return [:, -1, :]
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
    # no seq_length ?? batch_fist not true ??
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
    ### ??? !! batch_first: inputs.view(batch_size, -1, self.input_dim), self.hidden)
    ### ??? !! rnn_out return [:, -1, :]
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
    # no seq_length ?? batch_fist not true ??
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
        # ??? do I need this?
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
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # ?? need init_hidden()??
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
        ## ?? softmax loss
        if self.batch_first:
            x = F.sigmoid(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = F.sigmoid(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x

class ResNetLSTM(nn.Module):
    # no seq_length ?? batch_fist not true ??
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
        # ??? do I need this?
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
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # ?? need init_hidden()??
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
        ## ?? softmax loss
        if self.batch_first:
            x = F.sigmoid(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = F.sigmoid(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x

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

def train(model_name, rnn, gpu_id, train_loader, test_loader, criterion, optimizer, reg_method, reg_lambda, momentum_mu, blocks, n_hidden, weightdecay, firstepochs, labelnum, batch_first, n_epochs):
    logger = logging.getLogger('res_reg')
    res_regularizer_instance = ResRegularizer(reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=blocks, feature_dim=n_hidden)
    # Keep track of losses for plotting
    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    pre_running_loss = 0.0
    for epoch in range(n_epochs):
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
                
            if "reg" in model_name and epoch >= firstepochs:
                feature_idx = -1 # which feature to use for regularization
                for name, f in rnn.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (f.size())
                    if "layer1" in name and "weight_ih" in name:
                        # print ("check res_reg param name: ", name)
                        logger.debug ('res_reg param name: '+ name)
                        feature_idx = feature_idx + 1
                        res_regularizer_instance.apply(model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, len(train_loader.dataset), epoch, f, name, batch_idx, batch_first)
                        # print ("check len(train_loader.dataset): ", len(train_loader.dataset))
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
            running_loss += loss.item() * len(data_x)
            # print ('check!! len(data_x) --> last batch? : ', len(data_x))
            running_accuracy += accuracy * len(data_x)

        # Print epoch number, loss, name and guess
        # print ('maximum batch_idx: ', batch_idx)
        running_loss = running_loss / len(train_loader.dataset)
        running_accuracy = running_accuracy / len(train_loader.dataset)
        print('epoch: %d, training loss per sample per label =  %f, training accuracy =  %f'%(epoch, running_loss, running_accuracy))
        print('abs(running_loss - pre_running_loss)', abs(running_loss - pre_running_loss))
        pre_running_loss = running_loss

        # test
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
                print ('final test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(test_loss, accuracy, macro_auc, micro_auc))

    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print (do)
    elapsed = done - start
    print (elapsed)
    print('Finished Training')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Residual LSTM')
    parser.add_argument('-traindatadir', type=str, help='training data directory')
    parser.add_argument('-trainlabeldir', type=str, help='training label directory')
    parser.add_argument('-testdatadir', type=str, help='test data directory')
    parser.add_argument('-testlabeldir', type=str, help='test label directory')
    parser.add_argument('-seqnum', type=int, help='sequence number')
    parser.add_argument('-modelname', type=str, help='resnetrnn or reslstm or rnn or lstm')
    parser.add_argument('-blocks', type=int, help='number of blocks')
    parser.add_argument('-lr', type=float, help='0.001 for MIMIC-III')
    parser.add_argument('-decay', type=float, help='reg_lambda and weightdecay')
    parser.add_argument('-batchsize', type=int, help='batch_size')
    parser.add_argument('-regmethod', type=int, help='regmethod: : 0-calcRegGradAvg, 1-calcRegGradAvg_Exp, 2-calcRegGradAvg_Linear, 3-calcRegGradAvg_Inverse')
    parser.add_argument('-firstepochs', type=int, help='first epochs when no regularization is imposed')
    parser.add_argument('-considerlabelnum', type=int, help='just a reminder, need to consider label number because the loss is averaged across labels')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    # parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    parser.add_argument('--batch_first', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print ("args.debug: ", args.debug)
    print ("args.batch_first: ", args.batch_first)
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
    # train_x = np.genfromtxt(args.traindatadir, dtype=np.float32, delimiter=',')
    train_x_sparse_matrix = scipy.sparse.load_npz(args.traindatadir)
    train_x_sparse_matrix = train_x_sparse_matrix.astype(np.float32)
    train_x = np.array(train_x_sparse_matrix.todense())
    train_y = np.genfromtxt(args.trainlabeldir, dtype=np.float32, delimiter=',')
    train_y = train_y.reshape((train_y.shape[0],-1))
    # test_x = np.genfromtxt(args.testdatadir, dtype=np.float32, delimiter=',')
    test_x_sparse_matrix = scipy.sparse.load_npz(args.testdatadir)
    test_x_sparse_matrix = test_x_sparse_matrix.astype(np.float32)
    test_x = np.array(test_x_sparse_matrix.todense())
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


    n_hidden = 128
    n_epochs = args.maxepoch

    if "res" in args.modelname and "rnn" in args.modelname:
        print ('check resrnn model')
        rnn = ResNetRNN(args.gpuid, BasicResRNNBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
        rnn = rnn.cuda(args.gpuid)
    elif "res" in args.modelname and "lstm" in args.modelname:
        print ('check reslstm model')
        rnn = ResNetLSTM(args.gpuid, BasicResLSTMBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
        rnn = rnn.cuda(args.gpuid)
    elif "res" not in args.modelname and "rnn" in args.modelname:
        print ('check rnn model')
        rnn = ResNetRNN(args.gpuid, BasicRNNBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
        rnn = rnn.cuda(args.gpuid)
    elif "res" not in args.modelname and "lstm" in args.modelname:
        print ('check lstm model')
        rnn = ResNetLSTM(args.gpuid, BasicLSTMBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
        rnn = rnn.cuda(args.gpuid)
    else:
        print("Invalid model name, exiting...")
        exit()

    if "reg" in args.modelname:
        print ('optimizer without wd')
        optimizer = Adam(rnn.parameters(), lr=args.lr)
    else:
        print ('optimizer with wd')
        # optimizer = Adam(rnn.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer = optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    criterion = nn.BCELoss()
    momentum_mu = 0.9 # momentum mu
    reg_lambda = args.decay
    weightdecay = args.decay
    train(args.modelname, rnn, args.gpuid, train_loader, test_loader, criterion, optimizer, args.regmethod, reg_lambda, momentum_mu, args.blocks, n_hidden, weightdecay, args.firstepochs, label_num, args.batch_first, args.maxepoch)

####### real
# CUDA_VISIBLE_DEVICES=1 python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/sample/formal_valid_x_seq_sample.csv -trainlabel /hdd1/zhaojing/res-regularization/sample/formal_valid_y_seq_sample.csv -testdatadir /hdd1/zhaojing/res-regularization/sample/formal_valid_x_seq_sample.csv -testlabeldir /hdd1/zhaojing/res-regularization/sample/formal_valid_y_seq_sample.csv -seqnum 9 -modelname reslstm -blocks 2 -lr 0.001 -decay 0.00001 -batchsize 20 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 5 -gpuid 0 --batch_first --debug
# CUDA_VISIBLE_DEVICES=2 python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_x_seq_sample.csv -trainlabel /hdd1/zhaojing/res-regularization/sample/movie_review_valid_y_seq_sample.csv -testdatadir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_x_seq_sample.csv -testlabeldir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_y_seq_sample.csv -seqnum 25 -modelname resrnn -blocks 2 -lr 0.001 -decay 0.00001 -batchsize 20 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 2 -gpuid 0 --batch_first --debug
# CUDA_VISIBLE_DEVICES=0 python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_x_seq_sample.csv -trainlabel /hdd1/zhaojing/res-regularization/sample/movie_review_valid_y_seq_sample.csv -testdatadir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_x_seq_sample.csv -testlabeldir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_y_seq_sample.csv -seqnum 25 -modelname resmlp -blocks 2 -lr 0.08 -decay 0.00001 -batchsize 20 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 3 -gpuid 0 --debug | tee -a 2-14-check-mlp-movie-review
###############
# CUDA_VISIBLE_DEVICES=1 python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/sample/formal_valid_x_seq_sample.csv -trainlabel /hdd1/zhaojing/res-regularization/sample/formal_valid_y_seq_sample.csv -testdatadir /hdd1/zhaojing/res-regularization/sample/formal_valid_x_seq_sample.csv -testlabeldir /hdd1/zhaojing/res-regularization/sample/formal_valid_y_seq_sample.csv -seqnum 9 -modelname resrnn -blocks 2 -lr 0.001 -decay 0.00001 -batchsize 20 -regmethod 1 -firstepochs 3 -considerlabelnum 1 -maxepoch 5 -gpuid 0 --batch_first --debug
# CUDA_VISIBLE_DEVICES=0 python train_main_hook_resreg.py -datadir . -modelname rnn3 -blocks 2 -decay 0.00001 -regmethod 3 -firstepochs 0 -labelnum 1 -maxepoch 100000 -gpuid 0
# python train_hook_resreg.py regrnn3 0.005
# python train_hook.py rnn3 0.005
# python train_hook.py resrnn3 0.005
# python train_hook.py originrnn 0.005
