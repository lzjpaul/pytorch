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

import torch
import torch.nn as nn
from torch.autograd import Variable
from init_linear import InitLinear
from res_regularizer import ResRegularizer
import torch.autograd as autograd
from data import *
import random
import time
import math
import sys
import numpy as np
import logging
import argparse

features = []

class OriginRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OriginRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


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
        self.softmax1 = nn.LogSoftmax()

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
            x = self.softmax1(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = self.softmax1(self.fc1(x[-1, :, :].view(batch_size, -1)))
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

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

def train(model_name, rnn, gpu_id, criterion, optimizer, reg_method, reg_lambda, momentum_mu, blocks, n_hidden, weightdecay, firstepochs, labelnum, batch_first, n_epochs):
    logger = logging.getLogger('res_reg')
    res_regularizer_instance = ResRegularizer(reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=blocks, feature_dim=n_hidden)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = randomTrainingPair()
        if model_name != 'originrnn': 
            category_tensor, line_tensor = category_tensor.cuda(gpu_id), line_tensor.cuda(gpu_id)
        if batch_first:
            batch_size = line_tensor.size()[0]
        else:
            batch_size = line_tensor.size()[1]

        # output, loss = train(model_name, epoch, batch_size, batch_first, category_tensor, line_tensor, res_regularizer_instance)
        if model_name == 'originrnn':
            hidden = rnn.initHidden()
            optimizer.zero_grad()

            for i in range(line_tensor.size()[0]):
                output, hidden = rnn(line_tensor[i], hidden)

            loss = criterion(output, category_tensor)
            loss.backward()

            optimizer.step()
        else:
            rnn.init_hidden(batch_size)
            optimizer.zero_grad()
            features.clear()
            logger.debug ('line_tensor: ')
            logger.debug (line_tensor)
            logger.debug ('line_tensor norm: %f', line_tensor.norm())
            output = rnn(line_tensor)
            loss = criterion(output, category_tensor)

            logger.debug ("features length: %d", len(features))
            for feature in features:
                logger.debug ("feature size:")
                logger.debug (feature.data.size())
                logger.debug ("feature norm: %f", feature.data.norm())

            loss.backward()
            ### print norm
            for name, f in rnn.named_parameters():
                print ('param name: ', name)
                print ('param size:', f.data.size())
                print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                print ('lr 0.005 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 0.005))
            ### when to use res_reg
            
            if "reg" in model_name and epoch >= firstepochs:
                feature_idx = -1 # which feature to use for regularization
                for name, f in rnn.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (f.size())
                    if "layer1" in name and "weight_ih" in name:
                        logger.debug ('res_reg param name: '+ name)
                        feature_idx = feature_idx + 1
                        res_regularizer_instance.apply(gpu_id, features, feature_idx, 0, reg_lambda, labelnum, 800, epoch, f, name, epoch, batch_first)
                    else:
                        if weightdecay != 0:
                            logger.debug ('weightdecay name: ' + name)
                            logger.debug ('weightdecay: %f', weightdecay)
                            f.grad.data.add_(float(weightdecay), f.data)
                            logger.debug ('param norm: %f', np.linalg.norm(f.data.cpu().numpy()))
                            logger.debug ('weightdecay norm: %f', np.linalg.norm(float(weightdecay)*f.data.cpu().numpy()))
                            logger.debug ('lr 0.01 * param grad norm: %f', np.linalg.norm(f.grad.data.cpu().numpy() * 0.01))
            
            ### print norm
            optimizer.step()

        current_loss += loss.item()

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss.item(), line, guess, correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    print ('all_losses: ', all_losses)
    torch.save(rnn, 'char-rnn-classification.pt')

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Residual LSTM')
    parser.add_argument('-datadir', type=str, help='data directory')
    parser.add_argument('-modelname', type=str, help='resnetrnn or rnn')
    parser.add_argument('-blocks', type=int, help='number of blocks')
    parser.add_argument('-decay', type=float, help='reg_lambda and weightdecay')
    parser.add_argument('-regmethod', type=int, help='regmethod: : 0-calcRegGradAvg, 1-calcRegGradAvg_Exp, 2-calcRegGradAvg_Linear, 3-calcRegGradAvg_Inverse')
    parser.add_argument('-firstepochs', type=int, help='first epochs when no regularization is imposed')
    parser.add_argument('-labelnum', type=int, help='label number because the loss is averaged across labels')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    # parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    parser.add_argument('--batch_first', action='store_true')
    args = parser.parse_args()


    logging.basicConfig(level=logging.DEBUG, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger('res_reg')
    logger.info ('#################################')

    n_hidden = 128
    n_epochs = args.maxepoch
    print_every = 5000
    plot_every = 1000
    
    print ("args.batch_first: ", args.batch_first)

    if args.modelname == 'originrnn':
        rnn = OriginRNN(n_letters, n_hidden, n_categories)
    elif "res" in args.modelname:
        print ('resrnn model')
        rnn = ResNetRNN(args.gpuid, BasicResRNNBlock, n_letters, n_hidden, n_categories, args.blocks, args.batch_first)
        rnn = rnn.cuda(args.gpuid)
    else:
        print ('rnn model')
        rnn = ResNetRNN(args.gpuid, BasicRNNBlock, n_letters, n_hidden, n_categories, args.blocks, args.batch_first)
        rnn = rnn.cuda(args.gpuid)

    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.005)
    criterion = nn.NLLLoss()
    momentum_mu = 0.9 # momentum mu
    reg_lambda = args.decay
    weightdecay = args.decay
    train(args.modelname, rnn, args.gpuid, criterion, optimizer, args.regmethod, reg_lambda, momentum_mu, args.blocks, n_hidden, weightdecay, args.firstepochs, args.labelnum, args.batch_first, args.maxepoch)

# CUDA_VISIBLE_DEVICES=0 python train_main_hook_resreg.py -datadir . -modelname rnn3 -blocks 2 -decay 0.00001 -regmethod 3 -firstepochs 0 -labelnum 1 -maxepoch 100000 -gpuid 0
# python train_hook_resreg.py regrnn3 0.005
# python train_hook.py rnn3 0.005
# python train_hook.py resrnn3 0.005
# python train_hook.py originrnn 0.005
