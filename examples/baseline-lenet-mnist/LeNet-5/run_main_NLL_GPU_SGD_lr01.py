from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom
import onnx
import numpy as np
import torch.nn.functional as F

# viz = visdom.Visdom()
# cur_batch_win = None

def train(epoch, net, data_train_loader, optimizer, criterion):
    # global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    avg_train_loss = 0.0
    for i, (images, labels) in enumerate(data_train_loader):
        images = images.cuda(0)
        labels = labels.cuda(0)
        optimizer.zero_grad()

        output = net(images)

        # loss = criterion(output, labels)
        loss = F.nll_loss(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)
        avg_train_loss += (loss.detach().cpu().item() * images.shape[0])
        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
        """
        if viz.check_connection():
            cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)
        """

        loss.backward()
        ### print norm
        if i == 0:
            for name, f in net.named_parameters():
                print ('param name: ', name)
                print ('param size:', f.data.size())
                print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                print ('lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
        ### print norm
        optimizer.step()
    avg_train_loss /= 60000
    print('Train Avg. Loss: %f' % (avg_train_loss))

def test(net, data_test_loader, data_test, criterion):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        images = images.cuda(0)
        labels = labels.cuda(0)
        output = net(images)
        # avg_loss += criterion(output, labels).sum()  # still each sample mean
        avg_loss += F.nll_loss(output, labels).sum()  # still each sample mean
        # print ("criterion(output, labels): ", criterion(output, labels))
        # print ("criterion(output, labels).sum(): ", criterion(output, labels).sum())
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    print ("len(data_test): ", len(data_test))
    print ("len(data_test_loader): ", len(data_test_loader))
    avg_loss /= len(data_test_loader)
    print('Test Avg. Loss: %f, Accuracy: %f' % ((avg_loss.detach().cpu().item(), float(total_correct) / len(data_test))))


def train_and_test(epoch, net, data_train_loader, data_test_loader, data_test, optimizer, criterion):
    train(epoch, net, data_train_loader, optimizer, criterion)
    test(net, data_test_loader, data_test, criterion)

    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    dummy_input = dummy_input.cuda(0)
    # print ("before onnx export")
    torch.onnx.export(net, dummy_input, "lenet.onnx")

    onnx_model = onnx.load("lenet.onnx")
    # print ("before check model")
    # onnx.checker.check_model(onnx_model)
    print ("not check model??")


def main():
    data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
    data_test = MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    net = LeNet5()
    net = net.cuda(0)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(0)
    # optimizer = optim.Adam(net.parameters(), lr=2e-3)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    print ("len(data_train): ", len(data_train))
    """
    cur_batch_win_opts = {
        'title': 'Epoch Loss Trace',
        'xlabel': 'Batch Number',
        'ylabel': 'Loss',
        'width': 1200,
        'height': 600,
    }
    """
    # for e in range(1, 16):
    for e in range(1, 600):
        train_and_test(e, net, data_train_loader, data_test_loader, data_test, optimizer, criterion)


if __name__ == '__main__':
    main()
