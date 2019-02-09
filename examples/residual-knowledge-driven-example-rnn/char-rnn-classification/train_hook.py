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
## (1) batch_first set as argument
## (2) # of blocks: set as argument
## (3) batch_size are passed to the functions

import torch
import torch.nn as nn
from torch.autograd import Variable
from init_linear import InitLinear
import torch.autograd as autograd
from data import *
import random
import time
import math
import sys
import numpy as np

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
        print ('x norm: ', np.linalg.norm(x.data.cpu().numpy()))
        if self.batch_first:
            batch_size = x.size()[0]
            rnn_out, self.hidden = self.rnn(
                x.view(batch_size, -1, self.input_dim), self.hidden)
        else:
            batch_size = x.size()[1]
            rnn_out, self.hidden = self.rnn(
                x.view(-1, batch_size, self.input_dim), self.hidden)
        print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        print ('rnn_out/x ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(x.data.cpu().numpy()))
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
        print ('residual norm: ', np.linalg.norm(residual.data.cpu().numpy()))
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
        print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        print ('rnn_out/residual ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(residual.data.cpu().numpy()))
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

        # ??? do I need this?
        for idx, m in enumerate(self.modules()):
            # print ('idx and self.modules():')
            # print (idx)
            # print (m)
            if isinstance(m, nn.Conv2d):
                print ('initialization using kaiming_normal_')
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
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # ?? need init_hidden()??
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.rnn1(x)
        x = self.layer1(x)
        ## ?? softmax loss
        if self.batch_first:
            x = self.softmax1(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = self.softmax1(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

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

model_type = sys.argv[1]
gpu_id = 0
batch_first = False ## need to set this as argument ??? batch_first
learning_rate = float(sys.argv[2])
print ('learning_rate', learning_rate)
if model_type == 'originrnn':
    rnn = OriginRNN(n_letters, n_hidden, n_categories)
elif model_type == 'rnn3':
    blocks = 15
    rnn = ResNetRNN(gpu_id, BasicRNNBlock, n_letters, n_hidden, n_categories, blocks, batch_first)
    rnn = rnn.cuda(gpu_id)
elif model_type == 'resrnn3':
    blocks = 25
    rnn = ResNetRNN(gpu_id, BasicResRNNBlock, n_letters, n_hidden, n_categories, blocks, batch_first)
    rnn = rnn.cuda(gpu_id)
else:
    print ('Please specify one type model')

optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(model_type, batch_size, category_tensor, line_tensor):
    if model_type == 'originrnn':
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
        output = rnn(line_tensor)
        loss = criterion(output, category_tensor)

        loss.backward()
        ### print norm
        
        for name, f in rnn.named_parameters():
            print ('param name: ', name)
            print ('param size:', f.data.size())
            print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
            print ('lr 0.005 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 0.005))
        
        ### print norm
        optimizer.step()

    return output, loss.item()

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    if model_type != 'originrnn': 
        category_tensor, line_tensor = category_tensor.cuda(gpu_id), line_tensor.cuda(gpu_id)
    if batch_first:
        batch_size = line_tensor.size()[0]
    else:
        batch_size = line_tensor.size()[1]
    # print ('batch_size', batch_size)
    # print ('epoch: ', epoch)
    output, loss = train(model_type, batch_size, category_tensor, line_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

print ('all_losses: ', all_losses)
torch.save(rnn, 'char-rnn-classification.pt')
# python train_hook.py rnn3 0.005
# python train_hook.py resrnn3 0.005
# python train_hook.py originrnn 0.005
