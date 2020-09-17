# from lenet import LeNet5
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
from collections import OrderedDict
from res_regularizer_diff_dim import ResRegularizerDiffDim
import time
import datetime
import argparse
import logging
from baseline_method import BaselineMethod
# viz = visdom.Visdom()
# cur_batch_win = None

features = []

class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            # ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('c3', nn.Linear(400, 120)),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (img.data.size())
        logger.debug ('input norm: %f', img.data.norm())
        output = self.c3(img)
        logger.debug ('out size: ')
        logger.debug (output.data.size())
        logger.debug ('out norm: %f', output.data.norm())
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (img.data.size())
        logger.debug ('input norm: %f', img.data.norm())
        output = self.f4(img)
        logger.debug ('out size: ')
        logger.debug (output.data.size())
        logger.debug ('out norm: %f', output.data.norm())
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2() 
        self.c2_2 = C2() 
        self.c3 = C3()
        self.c3.register_forward_hook(get_features_hook)
        self.f4 = F4()
        self.f4.register_forward_hook(get_features_hook)
        self.f5 = F5()
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)
 
    def forward(self, img):
        # print ("img shape: ", img.shape)
        output = self.c1(img)
        # print ("output shape: ", output.shape)
        x = self.c2_1(output)
        # print ("x shape: ", x.shape)
        output = self.c2_2(output)
        # print ("output shape: ", output.shape)
        output += x
        # print ("after += output shape: ", output.shape)

        output = output.view(img.size(0), -1)
        # print ("before c3: ", output.shape)
        features.append(output.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check before blocks size:')
        logger.debug (output.data.size())
        logger.debug ('three models check before blocks norm: %f', output.data.norm())
        output = self.c3(output)
        logger.debug ('three models check after c3 size:')
        logger.debug (output.data.size())
        logger.debug ('three models check after c3 blocks norm: %f', output.data.norm())
        # print ("output shape: ", output.shape)
        # output = output.view(img.size(0), -1)
        # print ("output shape: ", output.shape)
        output = self.f4(output)
        # print ("output shape: ", output.shape)
        logger.debug ('three models check after blocks size:')
        logger.debug (output.data.size())
        logger.debug ('three models check after blocks norm: %f', output.data.norm())
        output = self.f5(output)
        # print ("final output shape: ", output.shape)
        return output

class DropoutLeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self, dropout):
        super(DropoutLeNet5, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2() 
        self.c2_2 = C2()
        self.drop3 = nn.Dropout(dropout) 
        self.c3 = C3()
        self.c3.register_forward_hook(get_features_hook)
        self.drop4 = nn.Dropout(dropout)
        self.f4 = F4()
        self.f4.register_forward_hook(get_features_hook)
        self.f5 = F5()
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)

    def forward(self, img):
        # print ("img shape: ", img.shape)
        output = self.c1(img)
        # print ("output shape: ", output.shape)
        x = self.c2_1(output)
        # print ("x shape: ", x.shape)
        output = self.c2_2(output)
        # print ("output shape: ", output.shape)
        output += x
        # print ("after += output shape: ", output.shape)

        output = output.view(img.size(0), -1)
        # print ("before c3: ", output.shape)
        output = self.drop3(output)
        features.append(output.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout3 data size:')
        logger.debug (output.data.size())
        logger.debug ('three models check after dropout3 data norm: %f', output.data.norm())
        output = self.c3(output)
        # print ("output shape: ", output.shape)
        # output = output.view(img.size(0), -1)
        # print ("output shape: ", output.shape)
        output = self.drop4(output)
        features.append(output.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout4 data size:')
        logger.debug (output.data.size())
        logger.debug ('three models check after dropout4 data norm: %f', output.data.norm())
        output = self.f4(output)
        # print ("output shape: ", output.shape)
        logger.debug ('three models check after blocks size:')
        logger.debug (output.data.size())
        logger.debug ('three models check after blocks norm: %f', output.data.norm())
        output = self.f5(output)
        # print ("final output shape: ", output.shape)
        return output


def get_features_hook(module, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    
    logger = logging.getLogger('res_reg')
    logger.debug('three models check Inside ' + module.__class__.__name__ + ' forward hook')
    logger.debug('')
    # logger.debug('input:')
    # logger.debug(input)
    logger.debug('three models check input: ')
    logger.debug(type(input))
    logger.debug('three models check input[0]: ')
    logger.debug(type(input[0]))
    logger.debug('three models check output: ')
    logger.debug(type(output))
    logger.debug('')
    logger.debug('three models check input[0] size:')
    logger.debug(input[0].size())
    logger.debug('three models check input norm: %f', input[0].data.norm())
    logger.debug('three models check output size:')
    logger.debug(output.data.size())
    logger.debug('three models check output norm: %f', output.data.norm())
    features.append(output.data)

def train(epoch, net, data_train_loader, optimizer, criterion, logger, res_regularizer_diff_dim_instance, model_name, firstepochs, labelnum,\
    baseline_method_instance, regmethod, lasso_strength, max_val):
    # global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    avg_train_loss = 0.0
    for i, (images, labels) in enumerate(data_train_loader):
        images = images.cuda(0)
        labels = labels.cuda(0)
        optimizer.zero_grad()

        features.clear()
        logger.debug ('three models check images: ')
        logger.debug (images)
        logger.debug ('three models check images shape: ')
        logger.debug (images.shape)

        output = net(images)

        # loss = criterion(output, labels)
        loss = F.nll_loss(output, labels)
        logger.debug ("three models check features length: %d", len(features))
        for feature in features:
            logger.debug ("three models check feature size:")
            logger.debug (feature.data.size())
            logger.debug ("three models check feature norm: %f", feature.data.norm())
        
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
        if epoch == 0 or i == 0:
            for name, f in net.named_parameters():
                print ('three models check param name: ', name)
                print ('three models check param size:', f.data.size())
                print ('three models check param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                print ('three models check lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
        ### when to use res_reg

        if "reg" in model_name:
            if regmethod == 6 and epoch >= firstepochs:
                feature_idx = -1 # which feature to use for regularization
            for name, f in net.named_parameters():
                logger.debug ("three models check param name: " +  name)
                logger.debug ("three models check param size:")
                logger.debug (f.size())
                if "c3.c3.c3.weight" in name or "f4.f4.f4.weight" in name:
                    if regmethod == 6 and epoch >= firstepochs:  # corr-reg
                        logger.debug ('three models check res_reg param name: '+ name)
                        feature_idx = feature_idx + 1
                        logger.debug ('three models check labelnum: %d', labelnum)
                        logger.debug ('three models check trainnum: %d', len(data_train_loader.dataset))
                        res_regularizer_diff_dim_instance.apply(model_name, 0, features, feature_idx, regmethod, reg_lambda, labelnum, 1, len(data_train_loader.dataset), epoch, f, name, i)
                        # res_regularizer_instance.apply(model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, seqnum, (train_data.size(0) * train_data.size(1))/seqnum, epoch, f, name, batch_idx, batch_first, cal_all_timesteps)
                        # print ("check len(train_loader.dataset): ", len(train_loader.dataset))
                    elif regmethod == 7:  # L1-norm
                        logger.debug ('L1 norm param name: '+ name)
                        logger.debug ('lasso_strength: %f', lasso_strength)
                        ### !! change param name to f ..
                        # print ("lasso param f: ", f)
                        baseline_method_instance.lasso_regularization(f, lasso_strength)
                    else:  # maxnorm and dropout
                        logger.debug ('no actions of param grad for maxnorm or dropout param name: '+ name)
                else:
                    if weightdecay != 0:
                        logger.debug ('three models check weightdecay name: ' + name)
                        logger.debug ('three models check weightdecay: %f', weightdecay)
                        f.grad.data.add_(float(weightdecay), f.data)
                        logger.debug ('three models check param norm: %f', np.linalg.norm(f.data.cpu().numpy()))
                        logger.debug ('three models check weightdecay norm: %f', np.linalg.norm(float(weightdecay)*f.data.cpu().numpy()))
                        logger.debug ('three models check lr 1.0 * param grad norm: %f', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))

        ### print norm
        optimizer.step()

        ### maxnorm constraist
        if "reg" in model_name and regmethod == 8:
            for name, param in net.named_parameters():  ##!!change model name!!
                logger.debug ("param name: " +  name)
                logger.debug ("param size:")
                logger.debug (param.size())
                if "c3.c3.c3.weight" in name or "f4.f4.f4.weight" in name:  ##!!change layer name!!
                    logger.debug ('max norm constraint for param name: '+ name)
                    logger.debug ('max_val: %f', max_val)
                    print ("max norm param param: ", param)
                    baseline_method_instance.max_norm(param, max_val)
        ### maxnorm constraist
    avg_train_loss /= 60000
    print('Train Avg. Loss: %f' % (avg_train_loss))

def test(net, data_test_loader, data_test, criterion, final=False):
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

    # print ("len(data_test): ", len(data_test))
    # print ("len(data_test_loader): ", len(data_test_loader))
    avg_loss /= len(data_test_loader)
    if final:
        print('final Test Avg. Loss: %f, Accuracy: %f' % ((avg_loss.detach().cpu().item(), float(total_correct) / len(data_test))))
    else:
        print('Test Avg. Loss: %f, Accuracy: %f' % ((avg_loss.detach().cpu().item(), float(total_correct) / len(data_test))))


def train_and_test(epoch, net, data_train_loader, data_test_loader, data_test, optimizer, criterion, \
    modelname, prior_beta, reg_lambda, momentum_mu, weightdecay, firstepochs, label_num, logger, res_regularizer_diff_dim_instance, \
    baseline_method_instance, regmethod, lasso_strength, max_val, final=False):
    # Keep track of losses for plotting

    train(epoch, net, data_train_loader, optimizer, criterion, logger, res_regularizer_diff_dim_instance, modelname, firstepochs, label_num, \
        baseline_method_instance, regmethod, lasso_strength, max_val)
    if final:
        print ('| final weightdecay {:.10f} | final prior_beta {:.10f} | final reg_lambda {:.10f} | final lasso_strength {:.10f} | final max_val {:.10f}'.format(weightdecay, prior_beta, reg_lambda, lasso_strength, max_val))
        test(net, data_test_loader, data_test, criterion, final=final)
    else:
        test(net, data_test_loader, data_test, criterion)

    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    dummy_input = dummy_input.cuda(0)
    # print ("before onnx export")
    # torch.onnx.export(net, dummy_input, "lenet.onnx")

    # onnx_model = onnx.load("lenet.onnx")
    # print ("before check model")
    # onnx.checker.check_model(onnx_model)
    print ("not check model??")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST LeNet')
    parser.add_argument('-modelname', type=str, help='reglenet or lenet')
    parser.add_argument('-firstepochs', type=int, help='first epochs when no regularization is imposed')
    parser.add_argument('-considerlabelnum', type=int, help='just a reminder, need to consider label number because the loss is averaged across labels')
    parser.add_argument('-regmethod', type=int, help='regmethod: : 6-corr-reg, 7-Lasso, 8-maxnorm, 9-dropout')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    # torch.manual_seed(args.seed)
    print ("args.debug: ", args.debug)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger('res_reg')
    logger.info ('#################################')

    data_train = MNIST('./lenet_data',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
    data_test = MNIST('./lenet_data',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    label_num = 1 # for MNIST
    print ("three models check label number: ", label_num)

    ########## using for
    # weightdecay_list = [0.0000001, 0.000001]
    # weightdecay_list = [0.0001]
    weightdecay_list = [0.0]
    reglambda_list = [1e-8]
    priorbeta_list = [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 200., 500., 1000.]
    lasso_strength_list = [1.0]
    max_val_list = [3.0]

    for weightdecay in weightdecay_list:
        for reg_lambda in reglambda_list:
            for prior_beta in priorbeta_list:
                for lasso_strength in lasso_strength_list:
                    for max_val in max_val_list:
                        print ('three models check weightdecay: ', weightdecay)
                        print ('three models check reg_lambda: ', reg_lambda)
                        print ('three models check priot prior_beta: ', prior_beta)
                        print ('lasso_strength: ', lasso_strength)
                        print ('max_val: ', max_val)
                        ########## using for
                        if "dropout" not in args.modelname:
                            net = LeNet5()
                        else:
                            net = DropoutLeNet5(args.dropout)
                        print ("net: ", net)
                        net = net.cuda(0)
                        criterion = nn.CrossEntropyLoss()
                        criterion = criterion.cuda(0)
                        # optimizer = optim.Adam(net.parameters(), lr=2e-3)
                        if "reg" in args.modelname:
                            print ('three models check optimizer without wd')
                            # optimizer = optim.SGD(net.parameters(), lr=0.05)
                            optimizer = optim.Adam(net.parameters(), lr=2e-3)
                        else:
                            print ('three models check optimizer with wd')
                            # optimizer = optim.SGD(net.parameters(), lr=0.05, weight_decay=weightdecay)
                            optimizer = optim.Adam(net.parameters(), lr=2e-3, weight_decay=weightdecay)
                        print ('three models check optimizer: ', optimizer)

                        print ("len(data_train): ", len(data_train))
                        # for e in range(1, 16):
                        momentum_mu = 0.9 # momentum mu

                        max_epoch = 200

                        logger = logging.getLogger('res_reg')
                        feature_dim_vec = [400, 120, 84]
                        res_regularizer_diff_dim_instance = ResRegularizerDiffDim(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=len(feature_dim_vec)-1, feature_dim_vec=feature_dim_vec, model_name=args.modelname)
                        baseline_method_instance = BaselineMethod()
                        start = time.time()
                        st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                        print(st)
                        for e in range(1, max_epoch):
                            if e != (max_epoch -1) and e != 100:
                                train_and_test(e, net, data_train_loader, data_test_loader, data_test, optimizer, criterion, args.modelname, prior_beta, \
                                    reg_lambda, momentum_mu, weightdecay, args.firstepochs, label_num, logger, res_regularizer_diff_dim_instance, baseline_method_instance, args.regmethod, lasso_strength, max_val)
                            else:  # last epoch ...
                                train_and_test(e, net, data_train_loader, data_test_loader, data_test, optimizer, criterion, args.modelname, prior_beta, \
                                    reg_lambda, momentum_mu, weightdecay, args.firstepochs, label_num, logger, res_regularizer_diff_dim_instance, baseline_method_instance, args.regmethod, lasso_strength, max_val, final=True)
                        done = time.time()
                        do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                        print (do)
                        elapsed = done - start
                        print (elapsed)
                        print('Finished Training')

# if __name__ == '__main__':
#     main()
