## what for:
## (1) using "for" for prior_beta, reg_lambda, weight_decay


## references:
# code for real-world healthcare and sentiment analysis
# https://github.com/lzjpaul/pytorch/blob/LDA-regularization/examples/cifar-10-tutorial/mimic_mlp_lda.py
# pytorch vision: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# resnet: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# MNIST dataset: https://github.com/pytorch/examples/blob/master/mnist/main.py
# inplace: https://blog.csdn.net/theonegis/article/details/81195065
# modify code for MIMIC-III
# https://github.com/lzjpaul/pytorch/blob/LDA-regularization/examples/cifar-10-tutorial/mimic_mlp.py
# https://github.com/lzjpaul/pytorch/blob/residual-knowledge-driven/examples/residual-knowledge-driven-example/mlp_residual_hook_resreg.py
# https://github.com/lalala16/mimic3-literatures-phenotyping_KDD/blob/master/method/test.py
######################################################################
# TODO
# 1) CrossEntropyLoss/BCELoss + softmax layer + metrix
# 2) mimic_metric (accuracy) --> MNIST?
# 3) optimizer --> healthcare??
# 4) Dataset
# 5) MNIST function runs for onece
# 6) adding seed for mini-batches?
# TODO-12-31
# 1) MyAdam for RNN?
# 2) set reg_lambda, weightdecay
# 3) which weights to be taken out? -- correct?
# Attention
# (1) label_num
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from init_linear import InitLinear
from res_regularizer import ResRegularizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse
import torchvision
from torchvision import datasets, models, transforms
import argparse
from mimic_metric import *
import time
import datetime
import logging
import torch.utils.data as Data
from torch.autograd import Variable

features = []

class BasicResMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BasicResMLPBlock, self).__init__()
        self.fc1 = InitLinear(input_dim, hidden_dim)

    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size: ')
        logger.debug (x.data.size())
        logger.debug ('inpit norm: %f', x.data.norm())
        residual = x
        out = F.sigmoid(self.fc1(x))
        out = out + residual
        logger.debug ('out size: ')
        logger.debug (out.data.size())
        logger.debug ('out norm: %f', out.data.norm())
        return out

class BasicMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BasicMLPBlock, self).__init__()
        self.fc1 = InitLinear(input_dim, hidden_dim)

    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('inpit norm: %f', x.data.norm())
        out = F.sigmoid(self.fc1(x))
        logger.debug ('out size: ')
        logger.debug (out.data.size())
        logger.debug ('out norm: %f', out.data.norm())
        return out


class ResNetMLP(nn.Module):
    def __init__(self, block, input_dim, hidden_dim, output_dim, blocks):
        super(ResNetMLP, self).__init__()
        self.fc1 = InitLinear(input_dim, hidden_dim)
        self.layer1 = self._make_layer(block, hidden_dim, hidden_dim, blocks)
        self.fc2 = InitLinear(hidden_dim, output_dim)

        logger = logging.getLogger('res_reg')
        # ??? do I need this?
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
            # elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, input_dim, hidden_dim, blocks):
        logger = logging.getLogger('res_reg')
        layers = []
        layers.append(block(input_dim, hidden_dim))
        for i in range(1, blocks):
            layers.append(block(input_dim, hidden_dim))
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        x = F.sigmoid(self.fc1(x))
        features.append(x.data)
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('before blocks norm: %f', x.data.norm())
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        x = F.sigmoid(self.fc2(x)) # dimension 0: # of samples, dimension 1: exponential
        return x

class MNISTResNetMLP(nn.Module):
    def __init__(self, block, input_dim, hidden_dim, output_dim, blocks):
        super(MNISTResNetMLP, self).__init__()
        self.fc1 = InitLinear(input_dim, hidden_dim)
        self.layer1 = self._make_layer(block, hidden_dim, hidden_dim, blocks)
        self.fc2 = InitLinear(hidden_dim, output_dim)

        logger = logging.getLogger('res_reg')
        # ??? do I need this?
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
            # elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, input_dim, hidden_dim, blocks):
        logger = logging.getLogger('res_reg')
        layers = []
        layers.append(block(input_dim, hidden_dim))
        for i in range(1, blocks):
            layers.append(block(input_dim, hidden_dim))
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        logger = logging.getLogger('res_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        x = F.sigmoid(self.fc1(x))
        features.append(x.data)
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('before blocks norm: %f', x.data.norm())
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        x = F.log_softmax(self.fc2(x), dim=1) # dimension 0: # of samples, dimension 1: exponential
        return x



def resnetmlp(blocks, dim_vec, pretrained=False, **kwargs):
    """Constructs a resnetmlp model.

    Args:
        blocks: how many residual links
        dim_vec: [input_dim, hidden_dim, output_dim]
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMLP(BasicResMLPBlock, dim_vec[0], dim_vec[1], dim_vec[2], blocks)
    return model

def mlp(blocks, dim_vec, pretrained=False, **kwargs):
    """Constructs a mlp model.

    Args:
        blocks: how many residual links
        dim_vec: [input_dim, hidden_dim, output_dim]
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMLP(BasicMLPBlock, dim_vec[0], dim_vec[1], dim_vec[2], blocks)
    return model

def mnistresnetmlp(blocks, dim_vec, pretrained=False, **kwargs):
    """Constructs a mnistresnetmlp model.

    Args:
        blocks: how many residual links
        dim_vec: [input_dim, hidden_dim, output_dim]
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MNISTResNetMLP(BasicResMLPBlock, dim_vec[0], dim_vec[1], dim_vec[2], blocks)
    return model

def mnistmlp(blocks, dim_vec, pretrained=False, **kwargs):
    """Constructs a mnistmlp model.

    Args:
        blocks: how many residual links
        dim_vec: [input_dim, hidden_dim, output_dim]
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MNISTResNetMLP(BasicMLPBlock, dim_vec[0], dim_vec[1], dim_vec[2], blocks)
    return model

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

def train_validate_test_resmlp_model(model_name, model, gpu_id, train_loader, test_loader, criterion, optimizer, reg_method, prior_beta, reg_lambda, momentum_mu, blocks, hidden_dim, weightdecay, firstepochs, labelnum, max_epoch=25):
    logger = logging.getLogger('res_reg')
    res_regularizer_instance = ResRegularizer(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=blocks, feature_dim=hidden_dim, model_name=model_name)
    # hyper parameters
    print('Beginning Training')
    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    pre_running_loss = 0.0
    for epoch in range(max_epoch):
        # Iterate over training data.
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for batch_idx, data_iter in enumerate(train_loader, 0):
            data_x, data_y = data_iter
            # print ("data_y shape: ", data_y.shape)
            data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
            logger.debug('data_x shape:')
            logger.debug(data_x.shape)
            logger.debug('data_y shape:')
            logger.debug(data_y.shape)
            optimizer.zero_grad()
            features.clear()
            logger.debug ('data_x: ')
            logger.debug (data_x)
            logger.debug ('data_x norm: %f', data_x.norm())
            outputs = model(data_x)
            loss = criterion(outputs, data_y)
            accuracy = AUCAccuracy(outputs.data.cpu().numpy(), data_y.data.cpu().numpy())[0]
            # train_loss += (loss.item() * len(data)) # sum over all samples in the mini-batch
            
            logger.debug ("features length: %d", len(features))
            for feature in features:
                logger.debug ("feature size:")
                logger.debug (feature.data.size())
                logger.debug ("feature norm: %f", feature.data.norm())
            
            loss.backward()
            ### print norm
            if (epoch == 0 and batch_idx < 1000) or batch_idx % 1000 == 0:
                for name, f in model.named_parameters():
                    print ('batch_idx: ', batch_idx)
                    print ('param name: ', name)
                    print ('param size:', f.data.size())
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                    print ('lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### when to use res-reg
            if "reg" in model_name and epoch >= firstepochs:
                feature_idx = -1 # which feature to use for regularization
                for name, param in model.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (param.size())
                    if "layer1" in name and "weight" in name:
                        # print ('check res_reg param name: ', name)
                        logger.debug ('res_reg param name: '+ name)
                        feature_idx = feature_idx + 1
                        res_regularizer_instance.apply(model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, 1, len(train_loader.dataset), epoch, param, name, batch_idx)
                    else:
                        if weightdecay != 0:
                            logger.debug ('weightdecay name: ' + name)
                            logger.debug ('weightdecay: %f', weightdecay)
                            param.grad.data.add_(float(weightdecay), param.data)
                            logger.debug ('param norm: %f', np.linalg.norm(param.data.cpu().numpy()))
                            logger.debug ('weightdecay norm: %f', np.linalg.norm(float(weightdecay)*param.data.cpu().numpy()))
                            logger.debug ('lr 1.0 * param grad norm: %f', np.linalg.norm(param.grad.data.cpu().numpy() * 1.0))
            ### print norm
            optimizer.step()
            running_loss += loss.item() * len(data_x)
            # print ('len(data_x): ', len(data_x))
            running_accuracy += accuracy * len(data_x)
            '''
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            '''
        # print ('maximum batch_idx: ', batch_idx)
        running_loss = running_loss / len(train_loader.dataset)
        running_accuracy = running_accuracy / len(train_loader.dataset)
        print('epoch: %d, training loss per sample per label =  %f, training accuracy =  %f'%(epoch, running_loss, running_accuracy))
        print('abs(running_loss - pre_running_loss)', abs(running_loss - pre_running_loss))
        pre_running_loss = running_loss

        # Iterate over test data.
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, data_iter in enumerate(test_loader):
                data_x, data_y = data_iter
                # print ("test data_y shape: ", data_y.shape)
                data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
                outputs = model(data_x)
                loss = criterion(outputs, data_y)
                metrics = AUCAccuracy(outputs.data.cpu().numpy(), data_y.data.cpu().numpy())
                accuracy, macro_auc, micro_auc = metrics[0], metrics[1], metrics[2]
                print ('test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(loss.item(), accuracy, macro_auc, micro_auc))
                if epoch == (max_epoch - 1):
                    print ('| final weightdecay {:.10f} | final prior_beta {:.10f} | final reg_lambda {:.10f}'.format(weightdecay, prior_beta, reg_lambda))
                    print ('final test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(loss.item(), accuracy, macro_auc, micro_auc))

    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print(do)
    elapsed = done - start
    print(elapsed)

def train_validate_test_resmlp_model_MNIST(model_name, model, gpu_id, train_loader, test_loader, criterion, optimizer, reg_method, prior_beta, reg_lambda, momentum_mu, blocks, hidden_dim, weightdecay, firstepochs, labelnum, max_epoch=25):
    logger = logging.getLogger('res_reg')
    res_regularizer_instance = ResRegularizer(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=blocks, feature_dim=hidden_dim, model_name=model_name)
    # hyper parameters
    print('Beginning Training')
    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    pre_train_loss = 0
    for epoch in range(max_epoch):
        # Iterate over training data.
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.reshape((data.shape[0],-1))
            logger.debug('data shape:')
            logger.debug(data.shape)
            # print ("target shape: ", target.shape)
            data, target = data.cuda(gpu_id), target.cuda(gpu_id)
            optimizer.zero_grad()
            features.clear()
            logger.debug ('data: ')
            logger.debug (data)
            logger.debug ('data norm: %f', data.norm())
            output = model(data)
            loss = F.nll_loss(output, target)
            train_loss += (loss.item() * len(data)) # sum over all samples in the mini-batch
            
            logger.debug ("features length: %d", len(features))
            for feature in features:
                logger.debug ("feature size:")
                logger.debug (feature.data.size())
                logger.debug ("feature norm: %f", feature.data.norm())
            
            loss.backward()
            ### print norm
            if (epoch == 0 and batch_idx <= 1000) or batch_idx % 1000 == 0:
                for name, f in model.named_parameters():
                    print ('batch_idx: ', batch_idx)
                    print ('param name: ', name)
                    print ('param size:', f.data.size())
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                    print ('lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### when to use res-reg
            if "reg" in model_name and epoch >= firstepochs:
                feature_idx = -1 # which feature to use for regularization
                for name, param in model.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (param.size())
                    if "layer1" in name and "weight" in name:
                        logger.debug ('res_reg param name: '+ name)
                        feature_idx = feature_idx + 1
                        res_regularizer_instance.apply(model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, 1, len(train_loader.dataset), epoch, param, name, batch_idx)
                    else:
                        if weightdecay != 0:
                            logger.debug ('weightdecay name: ' + name)
                            logger.debug ('weightdecay: %f', weightdecay)
                            param.grad.data.add_(float(weightdecay), param.data)
                            logger.debug ('param norm: %f', np.linalg.norm(param.data.cpu().numpy()))
                            logger.debug ('weightdecay norm: %f', np.linalg.norm(float(weightdecay)*param.data.cpu().numpy()))
                            logger.debug ('lr 1.0 * param grad norm: %f', np.linalg.norm(param.grad.data.cpu().numpy() * 1.0))
            ### print norm
            optimizer.step()
            '''
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            '''
        train_loss /= len(train_loader.dataset)
        print('epoch: ', epoch)
        print('train_loss per sample: ', train_loss)
        print('abs(train_loss - pre_train_loss)', abs(train_loss - pre_train_loss))
        pre_train_loss = train_loss

        # Iterate over test data.
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.reshape((data.shape[0],-1))
                data, target = data.cuda(gpu_id), target.cuda(gpu_id)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        if epoch == (max_epoch-1):
            print ('| final weightdecay {:.10f} | final prior_beta {:.10f} | final reg_lambda {:.10f}'.format(weightdecay, prior_beta, reg_lambda))
            print('\nfinal Test set: final Average loss: {:.4f}, final Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            
    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print(do)
    elapsed = done - start
    print(elapsed)

def initialize_model(model_name, blocks, dim_vec, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if "res" in model_name:
        """ resnetmlp or regresnetmlp
        """
        print ("resnetmlp")
        model_ft = resnetmlp(blocks, dim_vec, pretrained=use_pretrained)
    else:
        """ mlp or regmlp
        """
        print ("mlp")
        model_ft = mlp(blocks, dim_vec, pretrained=use_pretrained)
    # else:
    #     print("Invalid model name, exiting...")
    #     exit()
    
    return model_ft

def mnist_initialize_model(model_name, blocks, dim_vec, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if "res" in model_name:
        """ mnistresnetmlp or mnistregresnetmlp
        """
        print ("mnistresnetmlp")
        model_ft = mnistresnetmlp(blocks, dim_vec, pretrained=use_pretrained)
    else:
        """ mnistmlp or mnistregmlp
        """
        print ("mnistmlp")
        model_ft = mnistmlp(blocks, dim_vec, pretrained=use_pretrained)
    # else:
    #     print("Invalid model name, exiting...")
    #     exit()
    
    return model_ft

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Residual MLP')
    parser.add_argument('-traindatadir', type=str, help='training data directory')
    parser.add_argument('-trainlabeldir', type=str, help='training label directory')
    parser.add_argument('-testdatadir', type=str, help='test data directory')
    parser.add_argument('-testlabeldir', type=str, help='test label directory')
    parser.add_argument('-seqnum', type=int, help='sequence number')
    parser.add_argument('-modelname', type=str, help='resnetmlp or mlp')
    parser.add_argument('-blocks', type=int, help='number of blocks')
    parser.add_argument('-lr', type=float, help='0.08 for MIMIC-III, 0.01 for MNIST')
    parser.add_argument('-batchsize', type=int, help='batch_size, default 100, mnist hard-coded 64')
    parser.add_argument('-regmethod', type=int, help='regmethod: : 0-calcRegGradAvg, 1-calcRegGradAvg_Exp, 2-calcRegGradAvg_Linear, 3-calcRegGradAvg_Inverse')
    parser.add_argument('-firstepochs', type=int, help='first epochs when no regularization is imposed')
    parser.add_argument('-considerlabelnum', type=int, help='just a reminder, need to consider label number because the loss is averaged across labels')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    # parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    parser.add_argument('--batch_first', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
   
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
    if "MNIST" not in args.traindatadir:
        if "MIMIC-III" in args.traindatadir:
            train_x_sparse_matrix = scipy.sparse.load_npz(args.traindatadir)
            train_x_sparse_matrix = train_x_sparse_matrix.astype(np.float32)
            train_x = np.array(train_x_sparse_matrix.todense())
        else:
            train_x = np.genfromtxt(args.traindatadir, dtype=np.float32, delimiter=',')

        train_y = np.genfromtxt(args.trainlabeldir, dtype=np.float32, delimiter=',')
        train_y = train_y.reshape((train_y.shape[0],-1))
        # test_x = np.genfromtxt(args.testdatadir, dtype=np.float32, delimiter=',')
        if "MIMIC-III" in args.traindatadir:
            test_x_sparse_matrix = scipy.sparse.load_npz(args.testdatadir)
            test_x_sparse_matrix = test_x_sparse_matrix.astype(np.float32)
            test_x = np.array(test_x_sparse_matrix.todense())
        else:
            test_x = np.genfromtxt(args.testdatadir, dtype=np.float32, delimiter=',')

        test_y = np.genfromtxt(args.testlabeldir, dtype=np.float32, delimiter=',')
        test_y = test_y.reshape((test_y.shape[0],-1))
        if "MIMIC-III" in args.traindatadir:
            train_x = train_x.reshape((train_x.shape[0], args.seqnum, -1))
            test_x = test_x.reshape((test_x.shape[0], args.seqnum, -1))
            train_x = np.sum(train_x, axis=1, keepdims=False)
            test_x = np.sum(test_x, axis=1, keepdims=False)
        print ('check train_x.shape: ', train_x.shape)
        print ('check test_x.shape: ', test_x.shape)
        print ('check train_y.shape: ', train_y.shape)
        print ('check test_y.shape: ', test_y.shape)
        input_dim = train_x.shape[-1]
        print ('check input_dim: ', input_dim)

        train_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        train_loader = Data.DataLoader(dataset=train_dataset,
                                       batch_size=args.batchsize,
                                       shuffle=True)
        print ('len(train_dataset): ', len(train_dataset))
        test_dataset = Data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        test_loader = Data.DataLoader(dataset=test_dataset,
                                       batch_size=len(test_dataset),
                                       shuffle=True)
        print ('check!! len(test_dataset) !!: ', len(test_dataset))
    else:
        use_cuda = True 
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=64, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=1000, shuffle=True, **kwargs)

    print("Initializing Datasets and Dataloaders...")

    ########## using for
    weightdecay_list = [0.0000001, 0.000001]
    reglambda_list = [0.0002, 0.002]
    priorbeta_list = [0.0001, 0.001]

    for weightdecay in weightdecay_list:
        for reg_lambda in reglambda_list:
            for prior_beta in priorbeta_list:
                print ('weightdecay: ', weightdecay)
                print ('reg_lambda: ', reg_lambda)
                print ('priot prior_beta: ', prior_beta)
                ########## using for
                if "MNIST" not in args.traindatadir:
                    label_num = train_y.shape[1]
                    print ("check label number: ", label_num)
                    dim_vec = [input_dim, 128, label_num] # [input_dim, hidden_dim, output_num]
                    print ("check dim_vec: ", dim_vec)
                    

                    # Initialize the model for this run
                    model_ft = initialize_model(args.modelname, args.blocks, dim_vec, use_pretrained=False)
                else:
                    label_num = 1 # hard-coded for MNIST
                    print ("check label number: ", label_num)
                    dim_vec = [28*28, 100, 10] # [input_dim, hidden_dim, output_dim]
                    print ("check dim_vec: ", dim_vec)


                    model_ft = mnist_initialize_model(args.modelname, args.blocks, dim_vec, use_pretrained=False)
                # Print the model we just instantiated
                print('model:')
                print(model_ft)

                
                ######################################################################
                # Create the Optimizer
                # --------------------
                # Send the model to GPU
                model_ft = model_ft.cuda(gpu_id)

                # Gather the parameters to be optimized/updated in this run.
                params_to_update = model_ft.parameters()
                print("Params to learn:")
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        print("\t",name)
                
                # Observe that all parameters are being optimized
                if "reg" in args.modelname:
                    print ('optimizer without wd')
                    optimizer_ft = optim.SGD(params_to_update, lr=args.lr, momentum=0.9) ## correct for Helathcare or MNIST????
                    # optimizer_ft = optim.Adam(params_to_update, lr=args.lr) ## correct for Helathcare or MNIST????
                else:
                    print ('optimizer with wd')
                    # optimizer_ft = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
                    # optimizer_ft = optim.SGD(params_to_update, lr=args.lr, weight_decay=args.decay)
                    optimizer_ft = optim.SGD(params_to_update, lr=args.lr, momentum=0.9, weight_decay=weightdecay)
                    # optimizer_ft = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.decay)
                # optimizer_ft = optim.Adam(params_to_update, lr=0.01) ## correct for Helathcare or MNIST????

                ######################################################################
                # Run Training and Validation Step
                # --------------------------------
                #

                # Setup the loss fxn
                if "MNIST" not in args.traindatadir: 
                    criterion = nn.BCELoss() # ??? nn.loss or F.loss???
                    print("using BCELoss")
                else:
                    criterion = nn.CrossEntropyLoss() # ??? nn.loss or F.loss???
                    print("MNIST using CrossEntropyLoss")
                # Train and evaluate
                # train_validate_test_model(model_ft, gpu_id, train_loader, test_loader, criterion, optimizer_ft, max_epoch=args.maxepoch)
                momentum_mu = 0.9 # momentum mu
                # Train and evaluate MNIST on resmlp or mlp model
                if "MNIST" not in args.traindatadir: 
                    train_validate_test_resmlp_model(args.modelname, model_ft, gpu_id, train_loader, test_loader, criterion, optimizer_ft, args.regmethod, prior_beta, reg_lambda, momentum_mu, args.blocks, dim_vec[1], weightdecay, args.firstepochs, label_num, max_epoch=args.maxepoch)
                else:
                    train_validate_test_resmlp_model_MNIST(args.modelname, model_ft, gpu_id, train_loader, test_loader, criterion, optimizer_ft, args.regmethod, prior_beta, reg_lambda, momentum_mu, args.blocks, dim_vec[1], weightdecay, args.firstepochs, label_num, max_epoch=args.maxepoch)

# CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg_real_mnist.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 0 -modelname regmlp -blocks 2 -lr 0.01 -decay 0.00001 -reglambda 0.00001 -batchsize 65 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 5 -gpuid 0 --priorbeta 1.0
# CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg_real_mnist.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 0 -modelname mlp -blocks 2 -lr 0.01 -decay 0.00001 -reglambda 0.00001 -batchsize 65 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 --priorbeta 1.0
#CUDA_VISIBLE_DEVICES=0 python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabel /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname regmlp -blocks 1 -lr 0.3 -decay 0.00001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 50 -gpuid 0 --batch_first --priorbeta 1.5 --debug
# CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/sample/formal_valid_x_seq_sample.csv -trainlabel /hdd1/zhaojing/res-regularization/sample/formal_valid_y_seq_sample.csv -testdatadir /hdd1/zhaojing/res-regularization/sample/formal_valid_x_seq_sample.csv -testlabeldir /hdd1/zhaojing/res-regularization/sample/formal_valid_y_seq_sample.csv -seqnum 9 -modelname resmlp -blocks 2 -lr 0.08 -decay 0.00001 -batchsize 20 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 5 -gpuid 0 --debug
# CUDA_VISIBLE_DEVICES=0 python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_x_seq_sample.csv -trainlabel /hdd1/zhaojing/res-regularization/sample/movie_review_valid_y_seq_sample.csv -testdatadir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_x_seq_sample.csv -testlabeldir /hdd1/zhaojing/res-regularization/sample/movie_review_valid_y_seq_sample.csv -seqnum 25 -modelname mlp -blocks 2 -lr 0.08 -decay 0.00001 -batchsize 20 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 2 -gpuid 0 --debug
# CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/sample/formal_valid_x_seq_sample.csv -trainlabel /hdd1/zhaojing/res-regularization/sample/formal_valid_y_seq_sample.csv -testdatadir /hdd1/zhaojing/res-regularization/sample/formal_valid_x_seq_sample.csv -testlabeldir /hdd1/zhaojing/res-regularization/sample/formal_valid_y_seq_sample.csv -seqnum 9 -modelname resmlp -blocks 2 -lr 0.08 -decay 0.00001 -batchsize 20 -regmethod 1 -firstepochs 3 -considerlabelnum 1 -maxepoch 5 -gpuid 0 --debug
# CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 2 -decay 0.00001 -batchsize 64 -regmethod 5 -firstepochs 0 -labelnum 1 -maxepoch 200 -gpuid 0
# CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 1 -decay 0.00001 -batchsize 64 -maxepoch 10 -gpuid 0
# python mlp_residual_hook_resreg.py -datadir . -modelname regresnetmlp -blocks 3 -batchsize 64 -maxepoch 10 -gpuid 1
# python mlp_residual_hook_resreg.py -datadir . -modelname resnetmlp -blocks 3 -batchsize 64 -maxepoch 10 -gpuid 1
# python mlp_residual_hook_resreg.py -datadir . -modelname mlp -blocks 3 -batchsize 64 -maxepoch 10 -gpuid 1
