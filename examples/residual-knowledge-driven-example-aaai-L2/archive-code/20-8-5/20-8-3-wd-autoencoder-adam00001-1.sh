#### home folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-aaai-L2
##panda12 and panda2
### panda2
CUDA_VISIBLE_DEVICES=1 python autoencoder_MNIST_main_tune_wd_Adam00001_1.py -modelname autoenc -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-3/8-3-tune-wd-adam00001-1.log
