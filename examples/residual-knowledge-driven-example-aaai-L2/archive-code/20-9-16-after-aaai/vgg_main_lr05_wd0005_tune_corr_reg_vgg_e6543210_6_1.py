import argparse
import os
import shutil
import time
import datetime
import torch
import math

import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import vgg
import logging
from collections import OrderedDict
from res_regularizer import ResRegularizer
import numpy as np
from baseline_method import BaselineMethod

features = []

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'dropoutvgg16', 'dropoutvgg16_bn',
]

all_model_names = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'dropoutvgg16', 'dropoutvgg16_bn',
]

### relu(replace=True)
### https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd
### https://discuss.pytorch.org/t/the-inplace-operation-of-relu/40804/7
class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, featurelayers):
        super(VGG, self).__init__()
        self.featurelayers = featurelayers
        """
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        """
        # self.drop1 = nn.Dropout()
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU())
        ]))
        self.fc1.register_forward_hook(get_features_hook)
        # self.drop2 = nn.Dropout()
        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(512, 512)),
            ('relu2', nn.ReLU())
        ]))
        self.fc2.register_forward_hook(get_features_hook)
        self.fc3 = nn.Linear(512, 10)

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)



    def forward(self, x):
        logger = logging.getLogger('res_reg')
        x = self.featurelayers(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # x = self.drop1(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('three models check before blocks norm: %f', x.data.norm())
        x = self.fc1(x)
        logger.debug ('three models check after fc1 size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after fc1 norm: %f', x.data.norm())
        # x = self.drop2(x)
        x = self.fc2(x)
        logger.debug ('three models check after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after blocks norm: %f', x.data.norm())
        x = self.fc3(x)
        return x

class DropoutVGG(nn.Module):
    '''
    DropoutVGG model 
    '''
    def __init__(self, featurelayers, dropout):
        super(DropoutVGG, self).__init__()
        self.featurelayers = featurelayers
        """
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        """
        # self.drop1 = nn.Dropout()
        self.drop1 = nn.Dropout(dropout) 
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU())
        ]))
        self.fc1.register_forward_hook(get_features_hook)
        # self.drop2 = nn.Dropout()
        self.drop2 = nn.Dropout(dropout) 
        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(512, 512)),
            ('relu2', nn.ReLU())
        ]))
        self.fc2.register_forward_hook(get_features_hook)
        self.fc3 = nn.Linear(512, 10)

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)


    def forward(self, x):
        logger = logging.getLogger('res_reg')
        x = self.featurelayers(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # x = self.drop1(x)
        x = self.drop1(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout1 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout1 data norm: %f', x.data.norm())
        x = self.fc1(x)
        # x = self.drop2(x)
        x = self.drop2(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout2 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout2 data norm: %f', x.data.norm())
        x = self.fc2(x)
        logger.debug ('three models check after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after blocks norm: %f', x.data.norm())
        x = self.fc3(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))

def dropoutvgg16(dropout):
    """VGG 16-layer model (configuration "D") with dropout"""
    return DropoutVGG(make_layers(cfg['D']), dropout)

def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))

def dropoutvgg16_bn(dropout):
    """VGG 16-layer model (configuration "D") with batch normalization and dropout"""
    return DropoutVGG(make_layers(cfg['D'], batch_norm=True), dropout)

def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

"""
model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))
"""
model_names = sorted(name for name in all_model_names
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg"))


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
    logger.debug('three models check input tuple: ')
    logger.debug(input)
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


best_prec1 = 0

# if __name__ == '__main__':
def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-modelname', type=str, help='resnetrnn or reslstm or rnn or lstm')
    parser.add_argument('-firstepochs', type=int, help='first epochs when no regularization is imposed')
    parser.add_argument('-considerlabelnum', type=int, help='just a reminder, need to consider label number because the loss is averaged across labels')
    parser.add_argument('-regmethod', type=int, help='regmethod: : 6-corr-reg, 7-Lasso, 8-maxnorm, 9-dropout')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
    #                     choices=model_names,
    #                     help='model architecture: ' + ' | '.join(model_names) +
    #                     ' (default: vgg19)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
    #                     metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--cpu', dest='cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)

    global args, best_prec1
    args = parser.parse_args()

    print ("args.debug: ", args.debug)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger('res_reg')
    logger.info ('#################################')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./vgg_data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./vgg_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    label_num = 1 # for MNIST
    print ("three models check label number: ", label_num)

    ########## using for
    weightdecay_list = [5e-4]
    reglambda_list = [1e-1]
    priorbeta_list = [1e-4, 1e-3, 1e-2]
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
                        # Check the save_dir exists or not
                        if not os.path.exists(args.save_dir):
                            os.makedirs(args.save_dir)

                        if "vgg16" in args.modelname and "bn" not in args.modelname:
                            if "dropout" not in args.modelname:
                                model = vgg16()
                            else:
                                model = dropoutvgg16(args.dropout)
                        elif "vgg16" in args.modelname and "bn" in args.modelname:
                            if "dropout" not in args.modelname:
                                model = vgg16_bn()
                            else:
                                model = dropoutvgg16_bn(args.dropout)
                        # if "dropoutvgg16_bn" == args.arch:
                        #     model = dropoutvgg16_bn(args.dropout)
                        # elif "dropoutvgg16" == args.arch:
                        #     model = dropoutvgg16(args.dropout)
                        # elif "vgg16_bn" == args.arch:
                        #     model = vgg16_bn()
                        # elif "vgg16" == args.arch:
                        #     model = vgg16()
                        else:
                            print("Invalid model name, exiting...")
                            exit()

                        print ("model: ", model)

                        model.featurelayers = torch.nn.DataParallel(model.featurelayers)
                        if args.cpu:
                            model.cpu()
                        else:
                            model.cuda()

                        # optionally resume from a checkpoint
                        if args.resume:
                            if os.path.isfile(args.resume):
                                print("=> loading checkpoint '{}'".format(args.resume))
                                checkpoint = torch.load(args.resume)
                                args.start_epoch = checkpoint['epoch']
                                best_prec1 = checkpoint['best_prec1']
                                model.load_state_dict(checkpoint['state_dict'])
                                print("=> loaded checkpoint '{}' (epoch {})"
                                      .format(args.evaluate, checkpoint['epoch']))
                            else:
                                print("=> no checkpoint found at '{}'".format(args.resume))

                        cudnn.benchmark = True

                        # define loss function (criterion) and pptimizer
                        criterion = nn.CrossEntropyLoss()
                        if args.cpu:
                            criterion = criterion.cpu()
                        else:
                            criterion = criterion.cuda()

                        if args.half:
                            model.half()
                            criterion.half()

                        if "reg" in args.modelname:
                            print ('three models check optimizer without wd')
                            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                                        momentum=args.momentum)
                        else:
                            print ('three models check optimizer with wd')
                            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                                        momentum=args.momentum,
                                                        weight_decay=weightdecay)
                        print ('three models check optimizer: ', optimizer)
                       
                        momentum_mu = 0.9 # momentum mu
                        print ('three models check momentum_mu: ', momentum_mu) 
                        logger = logging.getLogger('res_reg')
                        res_regularizer_instance = ResRegularizer(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=2, feature_dim=512, model_name=args.modelname)
                        baseline_method_instance = BaselineMethod()
                        start = time.time()
                        st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                        print(st)

                        if args.evaluate:
                            validate(val_loader, model, criterion)
                            return

                        for epoch in range(args.start_epoch, args.epochs):
                            adjust_learning_rate(optimizer, epoch)

                            # train for one epoch
                            train(train_loader, model, criterion, optimizer, epoch, args.modelname, prior_beta, reg_lambda, momentum_mu, weightdecay, \
                                args.firstepochs, label_num, logger, res_regularizer_instance, baseline_method_instance, args.regmethod, lasso_strength, max_val)

                            # evaluate on validation set
                            if epoch != (args.epochs - 1):
                                prec1 = validate(val_loader, model, criterion)
                            else:  # last epoch
                                print ('| final weightdecay {:.10f} | final prior_beta {:.10f} | final reg_lambda {:.10f} | final lasso_strength {:.10f} | final max_val {:.10f}'.format(weightdecay, prior_beta, reg_lambda, lasso_strength, max_val))
                                prec1 = validate(val_loader, model, criterion, final=True)

                            # remember best prec@1 and save checkpoint
                            is_best = prec1 > best_prec1
                            best_prec1 = max(prec1, best_prec1)
                            save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_prec1': best_prec1,
                            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))
                        
                        done = time.time()
                        do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                        print (do)
                        elapsed = done - start
                        print (elapsed)
                        print('Finished Training')


def train(train_loader, model, criterion, optimizer, epoch, model_name, prior_beta, reg_lambda, momentum_mu, weightdecay, firstepochs, labelnum, \
    logger, res_regularizer_instance, baseline_method_instance, regmethod, lasso_strength, max_val):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cpu == False:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        if args.half:
            input = input.half()

        features.clear()
        logger.debug ('three models check input: ')
        logger.debug (input)
        logger.debug ('three models check input shape: ')
        logger.debug (input.shape)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        logger.debug ("three models check features length: %d", len(features))
        for feature in features:
            logger.debug ("three models check feature size:")
            logger.debug (feature.data.size())
            logger.debug ("three models check feature norm: %f", feature.data.norm())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        ### print norm
        if epoch == 0 or i % 1000 == 0:
            for name, f in model.named_parameters():
                print ('three models check param name: ', name)
                print ('three models check param size:', f.data.size())
                print ('three models check param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                print ('three models check lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
        ### when to use res_reg
        if "reg" in model_name:
            if regmethod == 6 and epoch >= firstepochs:
                feature_idx = -1 # which feature to use for regularization
            for name, f in model.named_parameters():
                logger.debug ("three models check param name: " +  name)
                logger.debug ("three models check param size:")
                logger.debug (f.size())
                if "fc1.fc1.weight" in name or "fc2.fc2.weight" in name:
                    if regmethod == 6 and epoch >= firstepochs:  # corr-reg
                        logger.debug ('three models check res_reg param name: '+ name)
                        feature_idx = feature_idx + 1
                        logger.debug ('three models check labelnum: %d', labelnum)
                        logger.debug ('three models check trainnum: %d', len(train_loader.dataset))
                        res_regularizer_instance.apply(model_name, 0, features, feature_idx, regmethod, reg_lambda, labelnum, 1, len(train_loader.dataset), epoch, f, name, i)
                        # res_regularizer_instance.apply(model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, seqnum, (train_data.size(0) * train_data.size(1))/seqnum, epoch, f, name, batch_idx, batch_first, cal_all_timesteps)
                        # print ("check len(train_loader.dataset): ", len(train_loader.dataset))
                    elif regmethod == 7:  # L1-norm
                        logger.debug ('L1 norm param name: '+ name)
                        logger.debug ('lasso_strength: %f', lasso_strength)
                        ### !! change param name to f ..
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
            for name, param in model.named_parameters():  ##!!change model name!!
                logger.debug ("param name: " +  name)
                logger.debug ("param size:")
                logger.debug (param.size())
                if "fc1.fc1.weight" in name or "fc2.fc2.weight" in name:  ##!!change layer name!!
                    logger.debug ('max norm constraint for param name: '+ name)
                    logger.debug ('max_val: %f', max_val)
                    baseline_method_instance.max_norm(param, max_val)
        ### maxnorm constraist

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]  # 0 means it is the top-1
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, final=False):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cpu == False:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    
    if final:
        print(' * final Prec@1 {top1.avg:.3f}'
              .format(top1=top1))
    else:
        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    # torch.save(state, filename)
    print ("disable saving models")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
