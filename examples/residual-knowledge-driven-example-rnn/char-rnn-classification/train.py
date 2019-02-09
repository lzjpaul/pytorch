## different from MIMIC-III
## https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
## Attention:
## (1) batch_first set as argument
## (2) # of blocks: set as argument

import torch
from data import *
from model import *
import random
import time
import math
import sys
import numpy as np

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
    print ('epoch: ', epoch)
    print ("features: ", features)
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
# python train.py rnn3 0.005
# python train.py resrnn3 0.005
# python train.py originrnn 0.005
