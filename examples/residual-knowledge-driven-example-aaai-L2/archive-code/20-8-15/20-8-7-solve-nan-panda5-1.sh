###### autoencoder-2, panda5-1
CUDA_VISIBLE_DEVICES=1 /hdd1/zhaojing/anaconda3-cuda-10/bin/python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_2.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-solve-nan-48.log
