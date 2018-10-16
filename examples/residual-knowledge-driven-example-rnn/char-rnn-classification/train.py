## different from MIMIC-III
## https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
## (1) batch_first

import torch
from data import *
from model import *
import random
import time
import math
import sys

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
gpu_id = 1
batch_first = False ## ??? batch_first
learning_rate = float(sys.argv[2])
if model_type == 'originrnn':
    rnn = OriginRNN(n_letters, n_hidden, n_categories)
elif model_type == 'rnn3':
    blocks = 3
    rnn = ResNetRNN(gpu_id, BasicRNNBlock, n_letters, n_hidden, n_categories, blocks, batch_first)
    rnn = rnn.cuda(gpu_id)
elif model_type == 'resrnn3':
    print ('not implemented yet')
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

torch.save(rnn, 'char-rnn-classification.pt')
# python train.py rnn3 0.005
# python train.py originrnn 0.005
