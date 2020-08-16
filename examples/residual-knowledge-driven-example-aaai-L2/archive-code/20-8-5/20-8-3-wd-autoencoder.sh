#### home folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-aaai-L2
##panda10
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python autoencoder_MNIST_main_tune_wd.py -modelname autoenc -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-3/8-3-tune-wd-1.log