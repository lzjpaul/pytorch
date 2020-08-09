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

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        #Encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256) # Input image (28*28 = 784)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)

        #Decoder 
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=784) # Output image (28*28 = 784)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))

        return x

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
def test_image_reconstruct(model, test_loader, device, criterion):
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
    print ("len(test_loader.dataset): ", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    print('Test Loss Per Sample: {:.3f}'.format(test_loss))

### The below function will be called to train the model. 
def training(model, train_loader, Epochs, test_loader, device, optimizer, criterion):
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
            outputs = model(img)
            loss = criterion(outputs, img)
            loss.backward()
            ### print norm
            if data_idx == 0:
                for name, f in model.named_parameters():
                    print ('param name: ', name)
                    print ('param size:', f.data.size())
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                    print ('lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### print norm
            optimizer.step()
            running_loss += loss.item()
            data_idx = data_idx + 1
            
        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, Epochs, loss))
        
        # evaluation
        test_image_reconstruct(model, test_loader, device, criterion)
        # evaluation

        if epoch % 5 == 0:
            save_decod_img(outputs.cpu().data, epoch)
    return train_loss

### The below function will test the trained model on image reconstruction.
## https://github.com/lzjpaul/pytorch/blob/residual-knowledge-driven/examples/residual-knowledge-driven-example-test-lda-prior/mlp_residual_hook_resreg_real_mnist_tune_hyperparam.py
"""
def test_image_reconstruct(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            img, _ = batch
            img = img.to(device)
            img = img.view(img.size(0), -1)
            outputs = model(img)
            print ("len(img): ", len(img))
            test_loss += (criterion(outputs, img).item() * len(img))
            # outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
            # save_image(outputs, 'MNIST_reconstruction.png')
            # break
    print ("len(test_loader.dataset): ", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    print('Test Loss Per Sample: {:.3f}'.format(test_loss))
"""

if __name__ == '__main__':
    Epochs = 200
    Lr_Rate = 1e-3
    Batch_Size = 128

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=Batch_Size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=Batch_Size, shuffle=True)

    print(train_set)

    # print(train_set.classes)
    model = Autoencoder()
    print(model)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)
    # optimizer = optim.SGD(model.parameters(), lr=10.0, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=10.0)
    ### Before training, the model will be pushed to the CUDA environment and the directory will be created to save the result images using the functions defined above.
    device = get_device()
    model.to(device)
    make_dir()

    ### Now, the training of the model will be performed.
    train_loss = training(model, train_loader, Epochs, test_loader, device, optimizer, criterion)

    ### image plot

    ### In the last step, we will test our autoencoder model to reconstruct the images.
    test_image_reconstruct(model, test_loader, device, criterion)

