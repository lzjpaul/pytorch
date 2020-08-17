### autoencoder, panda9-0
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python autoencoder_MNIST_main_SGD_lr10_wd000001_vis.py -modelname autoenc -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run/20-8-17-third-run-autoenc-wd.log
