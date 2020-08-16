## what for:
## (1) using "for" for prior_beta, reg_lambda, weight_decay


## references:
# 1) https://analyticsindiamag.com/hands-on-guide-to-implement-deep-autoencoder-in-pytorch-for-image-reconstruction/
######################################################################
# TODO
import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import argparse
from res_regularizer_diff_dim import ResRegularizerDiffDim
import time
import datetime
import logging
from collections import OrderedDict
from baseline_method import BaselineMethod

features = []

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        #Encoder
        # self.enc1 = nn.Linear(in_features=784, out_features=256) # Input image (28*28 = 784)
        self.enc1 = nn.Sequential(OrderedDict([
            ('enc1', nn.Linear(784, 256)),
            ('relu1', nn.ReLU())
        ]))
        # self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc2 = nn.Sequential(OrderedDict([
            ('enc2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU())
        ]))
        self.enc2.register_forward_hook(get_features_hook)
        # self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc3 = nn.Sequential(OrderedDict([
            ('enc3', nn.Linear(128, 64)),
            ('relu3', nn.ReLU())
        ]))
        self.enc3.register_forward_hook(get_features_hook)
        # self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc4 = nn.Sequential(OrderedDict([
            ('enc4', nn.Linear(64, 32)),
            ('relu4', nn.ReLU())
        ]))
        self.enc4.register_forward_hook(get_features_hook)
        # self.enc5 = nn.Linear(in_features=32, out_features=16)
        self.enc5 = nn.Sequential(OrderedDict([
            ('enc5', nn.Linear(32, 16)),
            ('relu5', nn.ReLU())
        ]))
        self.enc5.register_forward_hook(get_features_hook)

        #Decoder 
        # self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec1 = nn.Sequential(OrderedDict([
            ('dec1', nn.Linear(16, 32)),
            ('relu6', nn.ReLU())
        ]))
        self.dec1.register_forward_hook(get_features_hook)
        # self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec2 = nn.Sequential(OrderedDict([
            ('dec2', nn.Linear(32, 64)),
            ('relu7', nn.ReLU())
        ]))
        self.dec2.register_forward_hook(get_features_hook)
        # self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec3 = nn.Sequential(OrderedDict([
            ('dec3', nn.Linear(64, 128)),
            ('relu8', nn.ReLU())
        ]))
        self.dec3.register_forward_hook(get_features_hook)
        # self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec4 = nn.Sequential(OrderedDict([
            ('dec4', nn.Linear(128, 256)),
            ('relu9', nn.ReLU())
        ]))
        self.dec4.register_forward_hook(get_features_hook)
        # self.dec5 = nn.Linear(in_features=256, out_features=784) # Output image (28*28 = 784)
        self.dec5 = nn.Sequential(OrderedDict([
            ('dec5', nn.Linear(256, 784)),
            ('relu10', nn.ReLU())
        ]))
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)

    def forward(self, x):
        x = self.enc1(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check before blocks size:')
        logger.debug (x.data.size())
        # logger.debug ('three models check before blocks size: %s', x.data.size())
        logger.debug ('three models check before blocks norm: %f', x.data.norm())
        x = self.enc2(x)
        logger.debug ('three models check after enc2  size: %s', x.data.size())
        logger.debug ('three models check after enc2  norm: %f', x.data.norm())
        x = self.enc3(x)
        logger.debug ('three models check after enc3  size: %s', x.data.size())
        logger.debug ('three models check after enc3  norm: %f', x.data.norm())
        x = self.enc4(x)
        x = self.enc5(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        logger.debug ('three models check after dec3  size: %s', x.data.size())
        logger.debug ('three models check after dec3  norm: %f', x.data.norm())
        x = self.dec4(x)
        logger.debug ('three models check after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after blocks norm: %f', x.data.norm())
        x = self.dec5(x)

        return x

class DropoutAutoencoder(nn.Module):
    def __init__(self, dropout):
        super(DropoutAutoencoder, self).__init__()

        #Encoder
        # self.enc1 = nn.Linear(in_features=784, out_features=256) # Input image (28*28 = 784)
        self.enc1 = nn.Sequential(OrderedDict([
            ('enc1', nn.Linear(784, 256)),
            ('relu1', nn.ReLU())
        ]))
        self.drop2 = nn.Dropout(dropout)
        # self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc2 = nn.Sequential(OrderedDict([
            ('enc2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU())
        ]))
        self.enc2.register_forward_hook(get_features_hook)
        self.drop3 = nn.Dropout(dropout)
        # self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc3 = nn.Sequential(OrderedDict([
            ('enc3', nn.Linear(128, 64)),
            ('relu3', nn.ReLU())
        ]))
        self.enc3.register_forward_hook(get_features_hook)
        self.drop4 = nn.Dropout(dropout)
        # self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc4 = nn.Sequential(OrderedDict([
            ('enc4', nn.Linear(64, 32)),
            ('relu4', nn.ReLU())
        ]))
        self.enc4.register_forward_hook(get_features_hook)
        self.drop5 = nn.Dropout(dropout)
        # self.enc5 = nn.Linear(in_features=32, out_features=16)
        self.enc5 = nn.Sequential(OrderedDict([
            ('enc5', nn.Linear(32, 16)),
            ('relu5', nn.ReLU())
        ]))
        self.enc5.register_forward_hook(get_features_hook)

        #Decoder
        self.drop6 = nn.Dropout(dropout)
        # self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec1 = nn.Sequential(OrderedDict([
            ('dec1', nn.Linear(16, 32)),
            ('relu6', nn.ReLU())
        ]))
        self.dec1.register_forward_hook(get_features_hook)
        self.drop7 = nn.Dropout(dropout)
        # self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec2 = nn.Sequential(OrderedDict([
            ('dec2', nn.Linear(32, 64)),
            ('relu7', nn.ReLU())
        ]))
        self.dec2.register_forward_hook(get_features_hook)
        self.drop8 = nn.Dropout(dropout)
        # self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec3 = nn.Sequential(OrderedDict([
            ('dec3', nn.Linear(64, 128)),
            ('relu8', nn.ReLU())
        ]))
        self.dec3.register_forward_hook(get_features_hook)
        self.drop9 = nn.Dropout(dropout)
        # self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec4 = nn.Sequential(OrderedDict([
            ('dec4', nn.Linear(128, 256)),
            ('relu9', nn.ReLU())
        ]))
        self.dec4.register_forward_hook(get_features_hook)
        # self.dec5 = nn.Linear(in_features=256, out_features=784) # Output image (28*28 = 784)
        self.dec5 = nn.Sequential(OrderedDict([
            ('dec5', nn.Linear(256, 784)),
            ('relu10', nn.ReLU())
        ]))
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)


    def forward(self, x):
        x = self.enc1(x)
        x = self.drop2(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout2 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout2 data norm: %f', x.data.norm())
        x = self.enc2(x)
        x = self.drop3(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout3 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout3 data norm: %f', x.data.norm())
        x = self.enc3(x)
        x = self.drop4(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout4 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout4 data norm: %f', x.data.norm())
        x = self.enc4(x)
        x = self.drop5(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout5 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout5 data norm: %f', x.data.norm())
        x = self.enc5(x)

        x = self.drop6(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout6 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout6 data norm: %f', x.data.norm())
        x = self.dec1(x)
        x = self.drop7(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout7 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout7 data norm: %f', x.data.norm())
        x = self.dec2(x)
        x = self.drop8(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout8 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout8 data norm: %f', x.data.norm())
        x = self.dec3(x)
        x = self.drop9(x)
        features.append(x.data)
        logger.debug('three models check Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('three models check after dropout9 data size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after dropout9 data norm: %f', x.data.norm())
        x = self.dec4(x)
        logger.debug ('three models check after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('three models check after blocks norm: %f', x.data.norm())
        x = self.dec5(x)

        return x

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

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

### The below function will create a directory to save the results.
def make_dir():
    image_dir = 'MNIST_Out_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


### Using the below function, we will save the reconstructed images as generated by the model.
def save_decod_img(img, epoch):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, './MNIST_Out_Images/Autoencoder_image{}.png'.format(epoch))

### The below function will test the trained model on image reconstruction.
## https://github.com/lzjpaul/pytorch/blob/residual-knowledge-driven/examples/residual-knowledge-driven-example-test-lda-prior/mlp_residual_hook_resreg_real_mnist_tune_hyperparam.py
def test_image_reconstruct(model, test_loader, device, criterion, final=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            img, _ = batch
            img = img.to(device)
            img = img.view(img.size(0), -1)
            outputs = model(img)
            # print ("len(img): ", len(img))
            test_loss += (criterion(outputs, img).item() * len(img))
            # outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
            # save_image(outputs, 'MNIST_reconstruction.png')
            # break
    # print ("len(test_loader.dataset): ", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    if final:
        print('final Test Loss Per Sample: {:.3f}'.format(test_loss))
        print('final Test Loss All Samples: {:.3f}'.format(test_loss * len(test_loader.dataset)))
    else:
        print('Test Loss Per Sample: {:.3f}'.format(test_loss))
        print('Test Loss All Samples: {:.3f}'.format(test_loss * len(test_loader.dataset)))



### The below function will be called to train the model. 
def training(model, train_loader, Epochs, test_loader, device, optimizer, criterion, model_name, prior_beta, reg_lambda, momentum_mu, weightdecay, firstepochs, labelnum, regmethod, lasso_strength, max_val):
    logger = logging.getLogger('res_reg')
    feature_dim_vec = [256, 128, 64, 32, 16, 32, 64, 128, 256]
    res_regularizer_diff_dim_instance = ResRegularizerDiffDim(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=len(feature_dim_vec)-1, feature_dim_vec=feature_dim_vec, model_name=model_name)
    baseline_method_instance = BaselineMethod()
    # Keep track of losses for plotting
    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(st)

    train_loss = []
    for epoch in range(Epochs):
        model.train()
        running_loss = 0.0
        data_idx = 0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            features.clear()
            logger.debug ('three models check img: ')
            logger.debug (img)
            logger.debug ('three models check img shape: ')
            logger.debug (img.shape)
            outputs = model(img)
            loss = criterion(outputs, img)
            logger.debug ("three models check features length: %d", len(features))
            for feature in features:
                logger.debug ("three models check feature size:")
                logger.debug (feature.data.size())
                logger.debug ("three models check feature norm: %f", feature.data.norm())
            loss.backward()
            ### print norm
            if epoch == 0 or data_idx == 0:
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
                    if "enc2.enc2.weight" in name or "enc3.enc3.weight" in name or "enc4.enc4.weight" in name or "enc5.enc5.weight" in name \
                        or "dec1.dec1.weight" in name or "dec2.dec2.weight" in name or "dec3.dec3.weight" in name or "dec4.dec4.weight" in name:
                        if regmethod == 6 and epoch >= firstepochs:  # corr-reg
                            logger.debug ('three models check res_reg param name: '+ name)
                            feature_idx = feature_idx + 1
                            logger.debug ('three models check labelnum: %d', labelnum)
                            logger.debug ('three models check trainnum: %d', len(train_loader.dataset))
                            res_regularizer_diff_dim_instance.apply(model_name, 0, features, feature_idx, regmethod, reg_lambda, labelnum, 1, len(train_loader.dataset), epoch, f, name, data_idx)
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
                    if "enc2.enc2.weight" in name or "enc3.enc3.weight" in name or "enc4.enc4.weight" in name or "enc5.enc5.weight" in name \
                        or "dec1.dec1.weight" in name or "dec2.dec2.weight" in name or "dec3.dec3.weight" in name or "dec4.dec4.weight" in name:  ##!!change layer name!!
                        logger.debug ('max norm constraint for param name: '+ name)
                        logger.debug ('max_val: %f', max_val)
                        baseline_method_instance.max_norm(param, max_val)
            ### maxnorm constraist

            running_loss += loss.item()
            data_idx = data_idx + 1
            
        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, Epochs, loss))
        
        # evaluation
        if epoch != (Epochs-1):
            test_image_reconstruct(model, test_loader, device, criterion)
        else:  # last epoch
            print ('| final weightdecay {:.10f} | final prior_beta {:.10f} | final reg_lambda {:.10f} | final lasso_strength {:.10f} | final max_val {:.10f}'.format(weightdecay, prior_beta, reg_lambda, lasso_strength, max_val))
            test_image_reconstruct(model, test_loader, device, criterion, final=True)
        # evaluation

        if epoch % 5 == 0:
            save_decod_img(outputs.cpu().data, epoch)
    
    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print (do)
    elapsed = done - start
    print (elapsed)
    print('Finished Training')

    return train_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST autoencoder')
    parser.add_argument('-modelname', type=str, help='regautoenc or autoenc')
    parser.add_argument('-firstepochs', type=int, help='first epochs when no regularization is imposed')
    parser.add_argument('-considerlabelnum', type=int, help='just a reminder, need to consider label number because the loss is averaged across labels')
    parser.add_argument('-regmethod', type=int, help='regmethod: : 6-corr-reg, 7-Lasso, 8-maxnorm, 9-dropout')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print ("args.debug: ", args.debug)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger('res_reg')
    logger.info ('#################################')


    
    Epochs = 200
    Lr_Rate = 1e-3
    Batch_Size = 128

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(root='./autoencoder_data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./autoencoder_data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=Batch_Size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=Batch_Size, shuffle=True)

    print(train_set)

    label_num = 784  # just like the last layer output_dim is hardcoded ...
    print ("three models check label number: ", label_num)

    ########## using for
    # weightdecay_list = [0.0000001, 0.000001]
    weightdecay_list = [0.000001]
    reglambda_list = [1e-8, 1e-6, 1e-4, 1e-2, 1.]
    priorbeta_list = [1e-3, 1e-2, 1e-1, 1., 10., 100.]
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
                        # print(train_set.classes)
                        if "dropout" not in args.modelname:
                            model = Autoencoder()
                        else:
                            model = DropoutAutoencoder(args.dropout)
                        print(model)
                        criterion = nn.MSELoss()
                        # optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)
                        # optimizer = optim.SGD(model.parameters(), lr=10.0, momentum=0.9)
                        if "reg" in args.modelname:
                            print ('three models check optimizer without wd')
                            optimizer = optim.SGD(model.parameters(), lr=10.0)
                        else:
                            print ('three models check optimizer with wd')
                            optimizer = optim.SGD(model.parameters(), lr=10.0, weight_decay=weightdecay)
                        print ('three models check optimizer: ', optimizer)
                        ### Before training, the model will be pushed to the CUDA environment and the directory will be created to save the result images using the functions defined above.
                        device = get_device()
                        model.to(device)
                        make_dir()
                        momentum_mu = 0.9 # momentum mu
                        print ('three models check momentum_mu: ', momentum_mu)
                        ### Now, the training of the model will be performed.
                        train_loss = training(model, train_loader, Epochs, test_loader, device, optimizer, criterion, args.modelname, prior_beta, reg_lambda, momentum_mu, weightdecay, args.firstepochs, label_num, args.regmethod, lasso_strength, max_val)

                        ### image plot

                        ### In the last step, we will test our autoencoder model to reconstruct the images.
                        # test_image_reconstruct(model, test_loader, device, criterion)

