############################################################################ no dropout ##########################################################
#################################################################################################
### L2-reg (not using "reg" keyword because all layers using weight decay)
####### all are 7 * 6 = 42 params ...
###### autoencoder-1, panda5-1 (16 hours: 7)
### panda1-2
CUDA_VISIBLE_DEVICES=2 python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_autoencoder_lambda_1.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-1.log
############################################ 20-8-28-end !!!!!!!!!!!!!! ##############################################