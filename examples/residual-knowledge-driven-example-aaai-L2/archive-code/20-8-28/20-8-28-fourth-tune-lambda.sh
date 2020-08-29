############################################################################ no dropout ##########################################################
#################################################################################################
### L2-reg (not using "reg" keyword because all layers using weight decay)
####### all are 7 * 6 = 42 params ...
###### autoencoder-1, panda5-1 (16 hours: 7)
### panda1-2
CUDA_VISIBLE_DEVICES=2 python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_autoencoder_lambda_1.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-1.log

### panda2-1
CUDA_VISIBLE_DEVICES=1 python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_autoencoder_lambda_2.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-2.log

### panda2-2
CUDA_VISIBLE_DEVICES=2 python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_autoencoder_lambda_3.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-3.log

### panda3-0
CUDA_VISIBLE_DEVICES=0 python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_autoencoder_lambda_4.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-4.log

### panda3-2
CUDA_VISIBLE_DEVICES=2 python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_autoencoder_lambda_5.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-5.log

### panda4-0
CUDA_VISIBLE_DEVICES=0 python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_autoencoder_lambda_6.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-6.log

### panda4-1
CUDA_VISIBLE_DEVICES=1 python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_autoencoder_lambda_7.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-7.log

###### MLP-MIMIC-III, panda2-1 (16 hours: 33)
### panda4-2
CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg_real_mnist_wd00001_tune_corr_reg_mlp_mimic_iii_lambda_1.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname regmlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-8.log

### panda7-0
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real_mnist_wd00001_tune_corr_reg_mlp_mimic_iii_lambda_2.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname regmlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-9.log

############################################ 20-8-28-end !!!!!!!!!!!!!! ##############################################