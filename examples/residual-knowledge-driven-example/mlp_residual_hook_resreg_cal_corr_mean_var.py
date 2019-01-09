# refer to pytorch LDA MLP and resnet
# https://github.com/lzjpaul/pytorch/blob/LDA-regularization/examples/cifar-10-tutorial/mimic_mlp_lda.py
# pytorch vision: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# resnet: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# MNIST dataset: https://github.com/pytorch/examples/blob/master/mnist/main.py
# inplace: https://blog.csdn.net/theonegis/article/details/81195065
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

import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from init_linear import InitLinear
from res_regularizer import ResRegularizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import argparse
from mimic_metric import *
import time
import logging


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

def resnetmlp3(blocks, dim_vec, pretrained=False, **kwargs):
    """Constructs a resnetmlp3 model.

    Args:
        blocks: how many residual links
        dim_vec: [input_dim, hidden_dim, output_dim]
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMLP(BasicResMLPBlock, dim_vec[0], dim_vec[1], dim_vec[2], blocks)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def mlp3(blocks, dim_vec, pretrained=False, **kwargs):
    """Constructs a mlp3 model.

    Args:
        blocks: how many residual links
        dim_vec: [input_dim, hidden_dim, output_dim]
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMLP(BasicMLPBlock, dim_vec[0], dim_vec[1], dim_vec[2], blocks)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def get_features_hook(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    
    logger = logging.getLogger('res_reg')
    logger.debug('Inside ' + self.__class__.__name__ + ' forward hook')
    logger.debug('')
    logger.debug('input:')
    logger.debug(input)
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

'''
def train_validate_test_model(model, train_loader, test_loader, criterion, optimizer, max_epoch=25):
    
    since = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_accuracy = 0.0

        # Iterate over training data.
        for inputs, labels in train_loader:
            inputs = inputs.reshape((inputs.shape[0],-1))
            logger.debug('inputs shape:')
            logger.debug(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accuracy = AUCAccuracy(outputs, labels)[0] ## the metric may not be correct!! ??
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            running_accuracy += accuracy
            epoch_loss = running_loss / len(dataloaders[phase].dataset) # ???
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) # ???
            print ('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) # ???

        # Iterate over test data.
        for inputs, labels in test_loader:
            inputs = inputs.reshape((inputs.shape[0],-1))
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            metrics = AUCAccuracy(outputs, labels)
            test_accuracy, test_macro_auc, test_micro_auc = metrics[0], metrics[1], metrics[2] # ??? MNIST does not have ...
            # print statistics
            test_loss += loss.data[0]
            print ('test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(loss.data[0], accuracy, macro_auc, micro_auc))
            if epoch == (max_epoch - 1):
                print ('final test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(loss.data[0], accuracy, macro_auc, micro_auc))
'''

def train_validate_test_resmlp_model_MNIST(model_name, model, gpu_id, train_loader, test_loader, criterion, optimizer, reg_lambda, momentum_mu, hidden_dim, weightdecay, max_epoch=25):
    logger = logging.getLogger('res_reg')
    res_regularizer_instance = ResRegularizer(reg_lambda=reg_lambda, momentum_mu=momentum_mu, feature_dim=hidden_dim)
    # hyper parameters
    print('Beginning Training')
    since = time.time()
    
    for epoch in range(max_epoch):

        # Iterate over training data.
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.reshape((data.shape[0],-1))
            logger.debug('data shape:')
            logger.debug(data.shape)
            data, target = data.cuda(gpu_id), target.cuda(gpu_id)
            optimizer.zero_grad()
            features.clear()
            logger.debug ('data: ')
            logger.debug (data)
            logger.debug ('data norm: %f', data.norm())
            output = model(data)
            loss = F.nll_loss(output, target)
            
            logger.debug ("features length: %d", len(features))
            for feature in features:
                logger.debug ("feature size:")
                logger.debug (feature.data.size())
                logger.debug ("feature norm: %f", feature.data.norm())
            
            loss.backward()
            ### print norm
            ### when to use res-reg
            if "reg" in model_name:  
                logger.debug ('batch_idx %d', batch_idx) 
                for name, f in model.named_parameters():
                    logger.debug ('param name: ' + name)
                    logger.debug ('param size:')
                    logger.debug (f.data.size())
                    logger.debug ('param norm: %f', np.linalg.norm(f.data.cpu().numpy()))
                    logger.debug ('lr 0.01 * param grad norm: %f', np.linalg.norm(f.grad.data.cpu().numpy() * 0.01))
            
                feature_idx = -1 # which feature to use for regularization
                for name, param in model.named_parameters():
                    logger.debug ("param name: " +  name)
                    logger.debug ("param size:")
                    logger.debug (param.size())
                    if "layer1" in name and "weight" in name:
                        logger.debug ('res_reg param name: '+ name)
                        feature_idx = feature_idx + 1
                        res_regularizer_instance.apply(gpu_id, features, feature_idx, reg_lambda, epoch, param, name, batch_idx)
                    else:
                        if weightdecay != 0:
                            logger.debug ('weightdecay name: ' + name)
                            logger.debug ('weightdecay: %f', weightdecay)
                            param.grad.data.add_(float(weightdecay), param.data)
                            logger.debug ('param norm: %f', np.linalg.norm(param.data.cpu().numpy()))
                            logger.debug ('weightdecay norm: %f', np.linalg.norm(float(weightdecay)*param.data.cpu().numpy()))
                            logger.debug ('lr 0.01 * param grad norm: %f', np.linalg.norm(param.grad.data.cpu().numpy() * 0.01))
            ### print norm
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        
        # Iterate over train data to get correlation
        # if epoch == (max_epoch-1):
        if epoch >= 0:
            model.eval()
            verify_correlation_avg = np.zeros((hidden_dim, hidden_dim))
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_loader):
                    data = data.reshape((data.shape[0],-1))
                    data, target = data.cuda(gpu_id), target.cuda(gpu_id)
                    features.clear()
                    output = model(data)
                    # print("features length: ", len(features))
                    verify_correlation_avg = verify_correlation_avg + np.corrcoef(features[0].data.cpu().numpy(), features[1].data.cpu().numpy(), rowvar=False)[hidden_dim:, 0:hidden_dim]
            print ("batch_idx: ", batch_idx)
            verify_correlation_avg = verify_correlation_avg / float(batch_idx + 1)
            

            model.eval()
            verify_correlation_var_avg = np.zeros((hidden_dim, hidden_dim))
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_loader):
                    data = data.reshape((data.shape[0],-1))
                    data, target = data.cuda(gpu_id), target.cuda(gpu_id)
                    features.clear()
                    output = model(data)
                    # print("features length: ", len(features))
                    verify_correlation_var_avg = verify_correlation_var_avg + np.square(np.corrcoef(features[0].data.cpu().numpy(), features[1].data.cpu().numpy(), rowvar=False)[hidden_dim:, 0:hidden_dim] - verify_correlation_avg)
            print ("batch_idx: ", batch_idx)
            verify_correlation_var_avg = verify_correlation_var_avg / float(batch_idx + 1) 
            np.savetxt('verify_correlation_avg_matrix' + str(epoch), verify_correlation_avg, fmt = '%6f', delimiter=",") #modify here
            np.savetxt('verify_correlation_var_avg_matrix' + str(epoch), verify_correlation_var_avg, fmt = '%6f', delimiter=",") #modify here
                
        
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
            
    time_elapsed = time.time() - since

def initialize_model(model_name, blocks, dim_vec, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if "res" in model_name:
        """ resnetmlp3 or regresnetmlp3
        """
        print ("resnetmlp3")
        model_ft = resnetmlp3(blocks, dim_vec, pretrained=use_pretrained)
    elif model_name == "mlp3":
        """ mlp3
        """
        print ("mlp3")
        model_ft = mlp3(blocks, dim_vec, pretrained=use_pretrained)
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Residual MLP')
    parser.add_argument('-datadir', type=str, help='data directory')
    parser.add_argument('-modelname', type=str, help='resnetmlp3 or mlp3')
    parser.add_argument('-blocks', type=int, help='number of blocks')
    parser.add_argument('-decay', type=float, help='reg_lambda and weightdecay')
    parser.add_argument('-batchsize', type=int, help='batch_size')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    # parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    args = parser.parse_args()
   
    logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    gpu_id = args.gpuid
    logger = logging.getLogger('res_reg')
    logger.info ('#################################')
    print('gpu_id: ', gpu_id)

    # Initialize the model for this run
    dim_vec = [28*28, 100, 10] # [input_dim, hidden_dim, output_dim]
    model_ft = initialize_model(args.modelname, args.blocks, dim_vec, use_pretrained=False)

    # Print the model we just instantiated
    print('model:')
    print(model_ft)

    ######################################################################
    # Load Data
    # ---------
    #
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
        optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9) ## correct for Helathcare or MNIST????
    else:
        optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9, weight_decay=args.decay)
    # optimizer_ft = optim.Adam(params_to_update, lr=0.01) ## correct for Helathcare or MNIST????

    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    #

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss() # ??? nn.loss or F.loss???
    print("MNIST using CrossEntropyLoss")

    # Train and evaluate
    # train_validate_test_model(model_ft, gpu_id, train_loader, test_loader, criterion, optimizer_ft, max_epoch=args.maxepoch)
    momentum_mu = 0.9 # momentum mu
    reg_lambda = args.decay # resreg strength
    weightdecay = args.decay # other parameters' weight decay
    # Train and evaluate MNIST on resmlp or mlp model
    train_validate_test_resmlp_model_MNIST(args.modelname, model_ft, gpu_id, train_loader, test_loader, criterion, optimizer_ft, reg_lambda, momentum_mu, dim_vec[1], weightdecay, max_epoch=args.maxepoch)

# python mlp_residual_hook_resreg.py -datadir . -modelname regresnetmlp3 -blocks 3 -batchsize 64 -maxepoch 10 -gpuid 1
# python mlp_residual_hook_resreg.py -datadir . -modelname resnetmlp3 -blocks 3 -batchsize 64 -maxepoch 10 -gpuid 1
# python mlp_residual_hook_resreg.py -datadir . -modelname mlp3 -blocks 3 -batchsize 64 -maxepoch 10 -gpuid 1
